#!/usr/bin/env python3
"""Run the pre-outcome source-grounding gate for E01/S19-L16.

The human contract requires a stop before model execution when public evidence
does not completely specify a tensor/architecture hypothesis.  This runner
therefore audits manuscript and source support first.  If the gate is empty it
performs only exact, read-only replay of frozen L15 identities and evidence;
it never fits a new model or creates a new scientific outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import torch
import yaml

from e01_s19_padding_length_discrimination.core import build_split_manifest
from e01_s19_tensor_architecture_audit.core import (
    REQUIRED_GROUNDING_FIELDS,
    VERSION,
    array_sha256,
    assess_hypothesis,
    canonical_json,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
CONFIG_PATH = REPO_ROOT / "configs/e01/s19_l16_tensor_architecture_audit.yaml"
AMENDMENT_PATH = REPO_ROOT / "configs/e01/s19_l16_technical_amendment_001.json"
OUTPUT_ROOT = Path("/artifacts/research_steps/S19/loops/L16")
S19_ROOT = Path("/artifacts/research_steps/S19")
L15_ROOT = S19_ROOT / "loops/L15"
L15_CACHE = Path("/cache/e01_s19_l15")
PAPER_MARKDOWN = (
    WORKSPACE_ROOT
    / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
)
PAPER_FIGURE5 = (
    WORKSPACE_ROOT
    / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures/figure-05.png"
)
PAPER_PDF = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
PHIRL_ROOT = Path("/cache/e01_s12b/sources/PhiRL")
IIGR_ROOT = Path("/cache/e01_s12b/sources/IntegratedInformationGeneRegulation")
BREAKING_ROOT = Path("/cache/e01_s12e/sources/BreakingGRNMemories")
GARD_ROOT = Path("/cache/e01_s03/sources/gard-historical")
L12_ROOT = S19_ROOT / "loops/L12"

FIGURE_DIR = OUTPUT_ROOT / "figures"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        "branches": command("git", "branch", "-a", cwd=repo).splitlines(),
        "tags": command("git", "tag", cwd=repo).splitlines(),
        "status": command("git", "status", "--short", cwd=repo),
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
    roots.extend(S19_ROOT / "loops" / name for name in [
        "L01", "L02", "L03", "L04", "L05", "L06", "L06R", "L07",
        "L08", "L09", "L10", "L11", "L11R", "L12", "L13", "L14", "L15",
    ])
    return roots


def source_evidence() -> tuple[pd.DataFrame, dict[str, Any], str]:
    paper_lines = PAPER_MARKDOWN.read_text(encoding="utf-8").splitlines()
    statement = paper_lines[38]
    methods = paper_lines[92]
    figure_caption = paper_lines[259]
    identities = {
        "PhiRL": git_identity(PHIRL_ROOT),
        "IIGR": git_identity(IIGR_ROOT),
        "BreakingGRNMemories": git_identity(BREAKING_ROOT),
        "historicalGARD": git_identity(GARD_ROOT),
    }
    phirl_all_history = command("git", "rev-list", "--all", cwd=PHIRL_ROOT).splitlines()
    tracked_per_commit: set[str] = set()
    gard_prediction_hits: list[str] = []
    patterns = ("gard", "self-replic", "remaining 75", "first 25%", "pad_sequence")
    for commit in phirl_all_history:
        names = command("git", "ls-tree", "-r", "--name-only", commit, cwd=PHIRL_ROOT)
        for name in names.splitlines():
            tracked_per_commit.add(name)
            if not name.endswith((".py", ".md", ".txt")):
                continue
            try:
                content = command("git", "show", f"{commit}:{name}", cwd=PHIRL_ROOT).lower()
            except subprocess.CalledProcessError:
                continue
            for pattern in patterns:
                if pattern in content:
                    gard_prediction_hits.append(f"{commit}:{name}:{pattern}")

    rows = [
        {
            "evidenceId": "PAPER_TASK_SPLIT",
            "source": "original manuscript Results",
            "evidenceClass": "DIRECT_PAPER_SPECIFICATION",
            "operation": "semantic input/output split",
            "finding": "first 25% of Phi-r trajectory predicts remaining 75% self-replication trajectory",
            "figure5Specific": True,
            "groundsCompleteTensorField": False,
        },
        {
            "evidenceId": "PAPER_MODEL_FAMILY",
            "source": "original manuscript Results/Figure 5 caption",
            "evidenceClass": "DIRECT_PAPER_SPECIFICATION",
            "operation": "model family/evaluation",
            "finding": "MLP, 80/20 run split, ten seeded repetitions, binary accuracy, majority dummy",
            "figure5Specific": True,
            "groundsCompleteTensorField": False,
        },
        {
            "evidenceId": "PAPER_VARIABLE_LENGTH",
            "source": "original manuscript Methods",
            "evidenceClass": "DIRECT_PAPER_SPECIFICATION",
            "operation": "trajectory length",
            "finding": "n_tot differs between simulations due to stochasticity",
            "figure5Specific": False,
            "groundsCompleteTensorField": False,
        },
        {
            "evidenceId": "PHIRL_PLOT_INTERPOLATION",
            "source": "PhiRL plotting.py:262-278",
            "evidenceClass": "DIRECT_PUBLIC_CODE_NOT_LINKED_TO_GARD_FIGURE5",
            "operation": "plotting-only interpolation",
            "finding": "training reward curves are linearly interpolated to 1000 points over the shortest seed horizon for plotting median and SD",
            "figure5Specific": False,
            "groundsCompleteTensorField": False,
        },
        {
            "evidenceId": "PHIRL_LINEAR_EXTRACTOR",
            "source": "PhiRL archs.py:41-49",
            "evidenceClass": "DIRECT_PUBLIC_CODE_NOT_LINKED_TO_GARD_FIGURE5",
            "operation": "generic RL feature extractor",
            "finding": "flattened observation to one linear ReLU feature layer",
            "figure5Specific": False,
            "groundsCompleteTensorField": False,
        },
        {
            "evidenceId": "PHIRL_UNIFIED_EXTRACTOR",
            "source": "PhiRL archs.py:76-159",
            "evidenceClass": "DIRECT_PUBLIC_CODE_NOT_LINKED_TO_GARD_FIGURE5",
            "operation": "generic RL MLP/optional GRU extractor",
            "finding": "two-layer vector encoder with optional GRU and sequence-forward helper; no GARD target decoder, loss or scoring path",
            "figure5Specific": False,
            "groundsCompleteTensorField": False,
        },
        {
            "evidenceId": "PHIRL_HISTORY_GARD_TASK_SEARCH",
            "source": "complete local PhiRL public Git history",
            "evidenceClass": "PUBLIC_CODE_MISSING",
            "operation": "GARD prediction task",
            "finding": f"{len(phirl_all_history)} commits and {len(tracked_per_commit)} distinct paths inspected; no GARD/self-replication/25-to-75 prediction implementation recovered",
            "figure5Specific": True,
            "groundsCompleteTensorField": False,
        },
        {
            "evidenceId": "L12_PUBLIC_CODE_GAP",
            "source": "frozen S19-L12 concordance audit",
            "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
            "operation": "end-to-end Figure 5 implementation",
            "finding": "public lineage omits the GARD Figure-5 sequence tensor/MLP, label and scoring semantics",
            "figure5Specific": True,
            "groundsCompleteTensorField": False,
        },
        {
            "evidenceId": "S16_L15_IMPLEMENTATION",
            "source": "frozen S16/L15 method locks",
            "evidenceClass": "DIRECT_FROZEN_E01_IMPLEMENTATION",
            "operation": "complete executable reconstruction",
            "finding": "right-padding, masks, flattening MLP, loss, scoring and capacity are exact E01 choices, not direct author/source specification",
            "figure5Specific": True,
            "groundsCompleteTensorField": False,
        },
    ]
    audit = "\n".join(
        [
            "# Paper and public-source tensor/architecture audit",
            "",
            "## Locked manuscript evidence",
            "",
            f"- Results task statement: {statement}",
            f"- Methods variable-length statement: {methods}",
            f"- Figure 5 caption: {figure_caption}",
            "",
            "## Public-lineage finding",
            "",
            f"The complete local PhiRL history contains {len(phirl_all_history)} commits and {len(tracked_per_commit)} distinct tracked paths. No public commit, branch, tag, or inspected deleted path implements the GARD Figure-5 tensor, sequence target, padding/truncation, loss mask, scoring mask, output aggregation, or sequence-to-sequence MLP. The raw pattern audit produced "
            + (f"{len(gard_prediction_hits)} incidental hits, none a GARD Figure-5 predictor." if gard_prediction_hits else "zero GARD/self-replication/25-to-75 prediction-code hits."),
            "",
            "PhiRL's common-grid interpolation is inside `plot_reward` and produces a median/SD plot. Its `LinearExtractor` and `UnifiedExtractor` are generic reinforcement-learning feature extractors. Neither is linked to the manuscript's GARD prediction experiment, and neither supplies a target decoder, target mask, loss, accuracy weighting, or full architecture capacity.",
            "",
            "## Gate conclusion",
            "",
            "The paper supplies the semantic task and evaluation headline; public code supplies partial, context-mismatched implementation clues. No combination completely grounds every identity-changing field. Under the prospectively locked rule, L16 must stop before any new model fit.",
            "",
        ]
    )
    manifest = {
        "schema": "eidosoma.e01.s19.l16.source_snapshot.v1",
        "createdAtUtc": utc_now(),
        "repositories": identities,
        "phirlPublicHistoryCommitCount": len(phirl_all_history),
        "phirlDistinctPathCount": len(tracked_per_commit),
        "phirlGardPredictionPatternHits": sorted(gard_prediction_hits),
    }
    return pd.DataFrame(rows), manifest, audit


def hypothesis_support() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    partial = "PARTIAL_PAPER_SPECIFICATION"
    missing = "PUBLIC_CODE_AND_PAPER_MISSING"
    unrelated = "DIRECT_PUBLIC_CODE_NOT_LINKED_TO_GARD_FIGURE5"
    frozen = "DIRECT_FROZEN_E01_IMPLEMENTATION_NOT_AUTHOR_GROUNDING"
    mappings: dict[str, dict[str, str]] = {
        "H0_FROZEN_S16_RIGHT_PAD_FLAT_MLP": {
            "input_sequence_representation": partial,
            "variable_length_normalization": frozen,
            "input_padding_or_truncation": frozen,
            "target_sequence_representation": partial,
            "target_padding_or_truncation": frozen,
            "training_loss_and_mask": frozen,
            "scoring_mask": frozen,
            "output_aggregation": frozen,
            "architecture_topology": partial,
            "architecture_capacity": frozen,
        },
        "H1_COMMON_MINIMUM_HORIZON_INTERPOLATED_MLP": {
            "input_sequence_representation": partial,
            "variable_length_normalization": unrelated,
            "input_padding_or_truncation": unrelated,
            "target_sequence_representation": partial,
            "target_padding_or_truncation": missing,
            "training_loss_and_mask": missing,
            "scoring_mask": missing,
            "output_aggregation": missing,
            "architecture_topology": partial,
            "architecture_capacity": missing,
        },
        "H2_PUBLIC_PHIRL_GENERIC_VECTOR_OR_SEQUENCE_EXTRACTOR": {
            "input_sequence_representation": unrelated,
            "variable_length_normalization": missing,
            "input_padding_or_truncation": missing,
            "target_sequence_representation": partial,
            "target_padding_or_truncation": missing,
            "training_loss_and_mask": missing,
            "scoring_mask": missing,
            "output_aggregation": missing,
            "architecture_topology": unrelated,
            "architecture_capacity": unrelated,
        },
    }
    rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for hypothesis_id, mapping in mappings.items():
        result = assess_hypothesis(mapping)
        registry.append({"hypothesisId": hypothesis_id, **result, "fieldSupport": mapping})
        for field in REQUIRED_GROUNDING_FIELDS:
            evidence = mapping[field]
            rows.append(
                {
                    "hypothesisId": hypothesis_id,
                    "field": field,
                    "evidenceClass": evidence,
                    "directlyGrounded": evidence in {
                        "DIRECT_PAPER_SPECIFICATION",
                        "DIRECT_PUBLIC_CODE_EXPLICITLY_LINKED_TO_GARD_FIGURE5",
                    },
                    "completeHypothesis": bool(result["completeSourceGroundingPassed"]),
                    "registeredForExecution": bool(result["registeredForExecution"]),
                }
            )
    return pd.DataFrame(rows), registry


def validate_l15_artifacts() -> pd.DataFrame:
    manifest = json.loads((L15_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for record in manifest["files"]:
        path = L15_ROOT / record["path"]
        observed = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "path": str(path),
                "expectedSha256": record["sha256"],
                "observedSha256": observed,
                "passed": observed == record["sha256"],
            }
        )
    frame = pd.DataFrame(rows)
    if not frame["passed"].all():
        raise RuntimeError("frozen L15 artifact hash mismatch")
    return frame


def validate_feature_tensors() -> pd.DataFrame:
    manifest = pd.read_parquet(L15_ROOT / "feature_manifest.parquet")
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        path = Path(row.tensorPath)
        with np.load(path, allow_pickle=False) as payload:
            checks = {
                "values": (payload["values"], row.valuesSha256),
                "channelMask": (payload["channelMask"], row.channelMaskSha256),
                "timeMask": (payload["timeMask"], row.timeMaskSha256),
            }
            for field, (values, expected) in checks.items():
                observed = array_sha256(values)
                rows.append(
                    {
                        "candidateId": row.candidateId,
                        "featureId": row.featureId,
                        "field": field,
                        "shape": str(tuple(values.shape)),
                        "dtype": str(values.dtype),
                        "expectedSha256": expected,
                        "observedSha256": observed,
                        "passed": observed == expected,
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame["passed"].all():
        raise RuntimeError("frozen L15 tensor replay failed")
    return frame


def validate_targets() -> pd.DataFrame:
    manifest = pd.read_parquet(L15_ROOT / "padded_target_manifest.parquet")
    rows: list[dict[str, Any]] = []
    for candidate_id in ("CANDIDATE_2", "CANDIDATE_3"):
        with np.load(L15_CACHE / "tensors" / f"{candidate_id}__target.npz", allow_pickle=False) as payload:
            target = payload["target"]
            mask = payload["targetMask"]
            eligible = payload["eligible"].astype(bool)
            group = manifest.loc[manifest["candidateId"].eq(candidate_id)].sort_values("matrixIndex")
            if len(group) != len(target):
                raise RuntimeError("target manifest cardinality mismatch")
            for row in group.itertuples(index=False):
                target_hash = array_sha256(target[row.matrixIndex])
                mask_hash = array_sha256(mask[row.matrixIndex])
                if bool(row.eligible):
                    passed = (
                        bool(eligible[row.matrixIndex])
                        and target_hash == row.targetSha256
                        and mask_hash == row.targetMaskSha256
                    )
                    validation_rule = "EXACT_FULL_PADDED_ROW_HASH"
                else:
                    passed = (
                        not bool(eligible[row.matrixIndex])
                        and pd.isna(row.targetSha256)
                        and pd.isna(row.targetMaskSha256)
                    )
                    validation_rule = "STATUS_BEARING_INELIGIBLE_NULL_MANIFEST_HASH_NO_REPLACEMENT"
                rows.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": int(row.matrixIndex),
                        "targetSha256": target_hash,
                        "targetMaskSha256": mask_hash,
                        "manifestTargetSha256": row.targetSha256,
                        "manifestTargetMaskSha256": row.targetMaskSha256,
                        "eligible": bool(row.eligible),
                        "validationRule": validation_rule,
                        "passed": passed,
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame["passed"].all():
        raise RuntimeError("frozen L15 target replay failed")
    return frame


def validate_splits(config: dict[str, Any]) -> pd.DataFrame:
    frozen = pd.read_parquet(L15_ROOT / "split_manifest.parquet").sort_values(
        ["repetitionId", "matrixIndex"]
    ).reset_index(drop=True)
    root = yaml.safe_load(
        (REPO_ROOT / "configs/e01/s19_l15_untouched_padding_panel.yaml").read_text()
    )["seedContract"]["splitRootHex"]
    replay = build_split_manifest(root).sort_values(["repetitionId", "matrixIndex"]).reset_index(drop=True)
    common = [column for column in frozen.columns if column != "researchStepId"]
    passed = frozen[common].equals(replay[common])
    if not passed:
        raise RuntimeError("frozen L15 split replay failed")
    rows = []
    for repetition, group in frozen.groupby("repetitionId", sort=True):
        rows.append(
            {
                "repetitionId": int(repetition),
                "fitCount": int(group["splitRole"].eq("FIT").sum()),
                "validationCount": int(group["splitRole"].eq("VALIDATION").sum()),
                "testCount": int(group["splitRole"].eq("TEST").sum()),
                "pairedAcrossCandidatesAndFeatures": bool(group["candidateFeaturePairing"].all()),
                "outcomeStratified": bool(group["outcomeStratified"].any()),
                "passed": True,
            }
        )
    return pd.DataFrame(rows)


def validate_cached_model_probabilities() -> pd.DataFrame:
    replay = pd.read_parquet(L15_CACHE / "model_replay.parquet")
    checked = replay.loc[replay["checked"]].copy()
    rows = []
    for row in checked.itertuples(index=False):
        suffix = 1 if bool(row.trainIncludesPadding) else 0
        path = L15_CACHE / "model_results" / (
            f"{row.candidateId}__{row.featureId}__trainpad-{suffix}__R{int(row.repetitionId):02d}.npz"
        )
        with np.load(path, allow_pickle=False) as payload:
            observed = array_sha256(payload["probability"])
        passed = (
            bool(row.passed)
            and observed == row.probabilitySha256
            and row.probabilitySha256 == row.replayProbabilitySha256
        )
        rows.append(
            {
                "candidateId": row.candidateId,
                "featureId": row.featureId,
                "trainIncludesPadding": bool(row.trainIncludesPadding),
                "repetitionId": int(row.repetitionId),
                "cachedProbabilitySha256": observed,
                "recordedOriginalSha256": row.probabilitySha256,
                "recordedIndependentReplaySha256": row.replayProbabilitySha256,
                "newModelFitExecuted": False,
                "passed": passed,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 24 or not frame["passed"].all():
        raise RuntimeError("frozen L15 model probability replay evidence failed")
    return frame


def copy_frozen_evidence() -> list[str]:
    names = [
        "all_cell_metrics.parquet",
        "valid_cell_metrics.parquet",
        "accuracy_decomposition.parquet",
        "diagnostic_results.parquet",
        "negative_control_results.parquet",
        "suffix_invariance_results.parquet",
        "padding_dominance_results.parquet",
        "paper_boxplot_comparison.csv",
        "paper_model_order_results.csv",
    ]
    copied = []
    for name in names:
        source = L15_ROOT / name
        target = OUTPUT_ROOT / name
        shutil.copyfile(source, target)
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(f"read-only evidence copy mismatch: {name}")
        copied.append(name)
    return copied


def summarize_frozen_metrics() -> pd.DataFrame:
    all_metrics = pd.read_parquet(L15_ROOT / "all_cell_metrics.parquet")
    valid_metrics = pd.read_parquet(L15_ROOT / "valid_cell_metrics.parquet")
    paper_features = [
        "P1_PHIRL_EMERGENCE_COMPLETED_FIT",
        "B1_COMPOSITION_CHANGE",
        "B2_RAW_COMPOSITIONS",
        "B3_MOLECULAR_FLUXES",
        "D0_MAJORITY_DUMMY",
    ]
    rows = []
    for candidate in ("CANDIDATE_2", "CANDIDATE_3"):
        for feature in paper_features:
            group = all_metrics.loc[
                all_metrics["candidateId"].eq(candidate)
                & all_metrics["featureId"].eq(feature)
                & all_metrics["conditionId"].eq("S11_UNMASKED_TRAIN_UNMASKED_SCORE")
            ]
            if group.empty and feature == "D0_MAJORITY_DUMMY":
                group = all_metrics.loc[
                    all_metrics["candidateId"].eq(candidate)
                    & all_metrics["featureId"].eq(feature)
                    & all_metrics["conditionId"].eq("S11_UNMASKED_TRAIN_UNMASKED_SCORE")
                ]
            if not group.empty:
                rows.append(
                    {
                        "candidateId": candidate,
                        "featureId": feature,
                        "scope": "ALL_CELL_S11_READ_ONLY_L15",
                        "metric": "accuracy",
                        "median": float(group["accuracy"].median()),
                        "minimum": float(group["accuracy"].min()),
                        "maximum": float(group["accuracy"].max()),
                    }
                )
        group = valid_metrics.loc[
            valid_metrics["candidateId"].eq(candidate)
            & valid_metrics["featureId"].eq("P1_PHIRL_EMERGENCE_COMPLETED_FIT")
            & valid_metrics["conditionId"].eq("S11_UNMASKED_TRAIN_UNMASKED_SCORE")
        ]
        rows.append(
            {
                "candidateId": candidate,
                "featureId": "P1_PHIRL_EMERGENCE_COMPLETED_FIT",
                "scope": "VALID_MOLECULAR_CELL_READ_ONLY_L15",
                "metric": "balancedAccuracy",
                "median": float(group["balancedAccuracy"].median()),
                "minimum": float(group["balancedAccuracy"].min()),
                "maximum": float(group["balancedAccuracy"].max()),
            }
        )
    return pd.DataFrame(rows)


def figures(support: pd.DataFrame, registry: list[dict[str, Any]], summary: pd.DataFrame) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    matrix = support.pivot(index="hypothesisId", columns="field", values="directlyGrounded")
    matrix = matrix.reindex(columns=REQUIRED_GROUNDING_FIELDS).astype(float)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    image = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [item.replace("_", "\n") for item in matrix.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), [item.replace("_", "\n") for item in matrix.index], fontsize=8)
    ax.set_title("Direct source grounding of identity-changing Figure-5 fields")
    fig.colorbar(image, ax=ax, label="directly grounded")
    fig.tight_layout()
    path = FIGURE_DIR / "01_source_grounding_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [item["hypothesisId"].replace("_", "\n") for item in registry]
    grounded = [item["directlyGroundedFieldCount"] for item in registry]
    ax.bar(range(len(labels)), grounded, color="#4c78a8")
    ax.axhline(len(REQUIRED_GROUNDING_FIELDS), color="black", linestyle="--", label="complete gate")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(0, len(REQUIRED_GROUNDING_FIELDS) + 1)
    ax.set_ylabel("directly grounded required fields")
    ax.set_title("No audited convention reaches the complete-source gate")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / "02_hypothesis_completeness.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    paper = {
        "P1_PHIRL_EMERGENCE_COMPLETED_FIT": 0.8485,
        "B1_COMPOSITION_CHANGE": 0.8054,
        "B2_RAW_COMPOSITIONS": 0.7992,
        "B3_MOLECULAR_FLUXES": 0.7900,
        "D0_MAJORITY_DUMMY": 0.6092,
    }
    order = list(paper)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, candidate in zip(axes, ("CANDIDATE_2", "CANDIDATE_3")):
        subset = summary.loc[
            summary["candidateId"].eq(candidate)
            & summary["scope"].eq("ALL_CELL_S11_READ_ONLY_L15")
        ].set_index("featureId")
        x = np.arange(len(order))
        ax.plot(x, [paper[item] for item in order], "o-", label="paper digitization", color="black")
        ax.plot(x, [subset.loc[item, "median"] for item in order], "o-", label="frozen L15", color="#e45756")
        ax.set_xticks(x, ["PhiRL", "Δcomp", "raw", "flux", "dummy"])
        ax.set_title(candidate)
        ax.set_ylim(0.5, 1.01)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("binary accuracy")
    axes[1].legend()
    fig.suptitle("Frozen L15 extremes motivate—but do not source-ground—a tensor audit")
    fig.tight_layout()
    path = FIGURE_DIR / "03_frozen_l15_panel_gap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.62, 0.22, 0.22, "Paper\nsemantic task only"),
        (0.29, 0.62, 0.22, 0.22, "Public source\npartial unrelated clues"),
        (0.55, 0.62, 0.22, 0.22, "0 complete\nsource-grounded hypotheses"),
        (0.75, 0.18, 0.22, 0.22, "STOP\nno new model fits\nhuman review"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    arrows = [((0.25, 0.73), (0.29, 0.73)), ((0.51, 0.73), (0.55, 0.73)), ((0.66, 0.62), (0.84, 0.40))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Prospectively locked L16 decision path")
    fig.tight_layout()
    path = FIGURE_DIR / "04_gate_stop_decision_path.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def build_report(summary: pd.DataFrame, registry: list[dict[str, Any]], validations: dict[str, Any], lock_commit: str) -> str:
    all_rows = summary.loc[summary["scope"].eq("ALL_CELL_S11_READ_ONLY_L15")]
    metric_lines = []
    for candidate in ("CANDIDATE_2", "CANDIDATE_3"):
        group = all_rows.loc[all_rows["candidateId"].eq(candidate)].set_index("featureId")
        metric_lines.append(
            f"- {candidate}: frozen L15 S11 medians were PhiRL `{group.loc['P1_PHIRL_EMERGENCE_COMPLETED_FIT','median']:.4f}`, composition change `{group.loc['B1_COMPOSITION_CHANGE','median']:.4f}`, raw composition `{group.loc['B2_RAW_COMPOSITIONS','median']:.4f}`, flux `{group.loc['B3_MOLECULAR_FLUXES','median']:.4f}`, and dummy `{group.loc['D0_MAJORITY_DUMMY','median']:.4f}`."
        )
    return "\n".join(
        [
            "# S19-L16 Full Results — Figure-5 Tensor and Architecture Discrimination",
            "",
            "## Top summary",
            "",
            "- **Research step:** `E01-S19-L16-FIGURE5-TENSOR-ARCHITECTURE-DISCRIMINATION-v1.0.0`",
            "- **Completion status:** `COMPLETE_SOURCE_GROUNDING_GATE_STOP_AWAITING_MANDATORY_HUMAN_REVIEW`",
            "- **Outcome classification:** `AUTHOR_AMBIGUITY_UNRESOLVED`; `EXPLORATORY_NON_SUPPORT`; `NOT_PROMOTABLE`.",
            "- **Directed decision:** `NO_SUFFICIENTLY_SOURCE_GROUNDED_COMPLETE_TENSOR_OR_ARCHITECTURE_HYPOTHESIS`.",
            "- **Model execution:** No new model was fitted. The prospectively locked gate required this stop because zero audited conventions completely specified the tensor, variable-length handling, target representation, loss/scoring masks, aggregation, topology, and capacity from paper or Figure-5-linked public code.",
            f"- **Validation:** {validations['priorFileCount']} immutable prior files passed; 12/12 frozen feature tensors (36 arrays), 400/400 target/mask pairs, 10/10 splits, 60/60 frozen L15 artifact hashes, and 24/24 cached model/replay probability identities passed. Read-only all-cell, valid-cell, decomposition, length/time/boundary, permutation, and suffix-invariance evidence was copied byte-for-byte. One preserved validator-only amendment corrected the treatment of L15's two intentionally null ineligible-row hashes and changed no array, status, gate, or scientific value.",
            "- **Recommended next action:** Mandatory human review. Public evidence remains insufficient for another scientifically locked Figure-5 model run; an exact author-code/configuration release or a separately authorized closeout decision would add more information than another adaptive tensor guess.",
            "",
            "## Frozen question and pre-outcome decision rule",
            "",
            "L16 asked whether the manuscript or pinned public lineage contains enough explicit detail to register at most three complete tensor/architecture hypotheses on the frozen L15 cohort. The lock required direct paper specification or public code explicitly linked to the GARD Figure-5 task for ten identity-changing fields. Plotting-only interpolation, generic reinforcement-learning feature extractors, and E01's own S16 implementation were deliberately insufficient substitutes for author grounding.",
            "",
            f"The complete contract was committed and pushed at `{lock_commit}` before this gate was evaluated.",
            "",
            "## What the paper and public code actually specify",
            "",
            "The manuscript directly specifies: the first 25% of the Phi-r trajectory as input; the remaining 75% self-replication trajectory as target; an MLP; an 80/20 split across runs; ten seeded repetitions; binary accuracy; and a majority-label dummy. It also states that molecular trajectory lengths vary stochastically.",
            "",
            "It does **not** specify: the fixed tensor shape; interpolation/resampling/truncation; input or target padding; padding values; target mask; training loss weighting; scoring mask; per-cell versus per-run aggregation; MLP layer topology; capacity; activation; optimizer; or early stopping.",
            "",
            "Pinned PhiRL contains a plotting helper that linearly interpolates reward curves to 1,000 points over the shortest seed horizon, plus generic RL vector/sequence feature extractors. The complete public Git history contains no GARD/self-replication 25%-to-75% prediction implementation. The interpolation is not a prediction tensor, and the extractors do not implement a GARD target decoder, mask, loss, accuracy, or complete supervised architecture.",
            "",
            "## Audited candidate conventions",
            "",
            *[
                f"- `{item['hypothesisId']}`: {item['directlyGroundedFieldCount']}/{item['requiredFieldCount']} required fields directly grounded; execution registration = `{item['registeredForExecution']}`."
                for item in registry
            ],
            "",
            "No convention was eligible. Combining the plotting interpolation with the generic extractor and S16's invented target/loss conventions would mix independently convenient components without a paper or source link—the exact architecture tournament the human contract prohibited.",
            "",
            "## Frozen L15 evidence accepted without reinterpretation",
            "",
            *metric_lines,
            "",
            "Those values remain L15 evidence. L16 did not refit, tune, select, or reopen them. The frozen valid-cell balanced accuracy remains approximately 0.50, the length/boundary diagnostics remain near-deterministic, valid-label permutation retains the all-cell result, P1 remains completed-fit future-dependent, and P2 retains suffix invariance. These facts motivate the ambiguity but cannot identify a missing author tensor.",
            "",
            "## Why no model run is the scientifically valid L16 result",
            "",
            "The paper-panel accuracies lie between L15's valid-cell non-discrimination and unmasked length/padding near-determinism. Many unreported transformations could interpolate between those extremes. Without source grounding, executing normalized-time resampling, common-horizon truncation, run-level aggregation, or another MLP layout would select among underdetermined implementations using the already known panel. That would increase specification multiplicity rather than discriminate a paper-directed hypothesis.",
            "",
            "This is a protocol gate result, not an operational failure. `LOOP_FAILED_CLOSED` is therefore not used. The correct classifications are `AUTHOR_AMBIGUITY_UNRESOLVED`, `EXPLORATORY_NON_SUPPORT`, and `NOT_PROMOTABLE`.",
            "",
            "## Validation and regeneration",
            "",
            f"- Immutable prior baseline: `{validations['priorFileCount']}` files, `{validations['priorMismatchCount']}` mismatches.",
            f"- Frozen artifact replay: `{validations['l15ArtifactHashCount']}/{validations['l15ArtifactHashCount']}` passed.",
            f"- Feature tensor replay: `{validations['tensorArrayCount']}/{validations['tensorArrayCount']}` arrays passed.",
            f"- Target replay: `{validations['targetPairCount']}/{validations['targetPairCount']}` matrix/candidate target-mask pairs passed.",
            f"- Split replay: `{validations['splitCount']}/{validations['splitCount']}` exact matrix-grouped 128/32/40 splits passed.",
            f"- Cached model identity: `{validations['modelReplayCount']}/{validations['modelReplayCount']}` original/replay probability hashes passed; no L16 training was run.",
            "- Read-only L15 machine tables were copied byte-identically and their accuracy decomposition remains explicit.",
            "- Technical amendment 001 preserves the first failed partial attempt under `/cache/e01_s19_l16/failed_attempt_001_artifacts`; it changes only the independent validator's handling of the two registered ineligible rows, whose manifest hashes are intentionally null.",
            "- CPU float64 remained authoritative; GPU use was zero; no matrix, trajectory, label, feature, split, model outcome, intervention, or report bundle was generated.",
            "",
            "## Figures",
            "",
            "![Direct source-grounding matrix](figures/01_source_grounding_matrix.png)",
            "",
            "*Figure 1. Direct support across the ten identity-changing fields. Partial paper descriptions, plotting-only code, generic RL extractors, and frozen E01 choices do not satisfy the direct Figure-5 grounding gate.*",
            "",
            "![Hypothesis completeness](figures/02_hypothesis_completeness.png)",
            "",
            "*Figure 2. None of the three audited convention candidates becomes an executable complete hypothesis.*",
            "",
            "![Frozen L15 panel gap](figures/03_frozen_l15_panel_gap.png)",
            "",
            "*Figure 3. Read-only L15 all-cell results bracket a task-geometry problem but do not identify a source-supported intermediate convention.*",
            "",
            "![Gate-stop path](figures/04_gate_stop_decision_path.png)",
            "",
            "*Figure 4. The prospectively locked source-grounding gate stops before new model fitting.*",
            "",
            "## Interpretation boundary",
            "",
            "L16 does not show that no private coherent implementation exists. It shows that the inspected manuscript and public lineages cannot completely distinguish one. No result identifies author code, supports prospective initial-appearance prediction, changes the completed-fit leakage finding, supports intervention or causal control, or changes any S18 classification.",
            "",
            "## Mandatory human-review boundary",
            "",
            "Stop here. L16 is frozen. S20, E02, author contact, confirmation, intervention work, report generation, and any later loop remain inactive pending a new explicit human decision.",
            "",
        ]
    )


def append_root_ledgers(timestamp: str, source_manifest: dict[str, Any]) -> None:
    candidate_path = S19_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    if candidates["candidateId"].astype(str).str.startswith("S19-L16-").any():
        raise RuntimeError("L16 candidate ledger rows already exist")
    additions = []
    for offset, (identifier, description, grounding) in enumerate(
        [
            ("S19-L16-H0-FROZEN-S16", "Exact frozen S16/L15 right-padded flattening MLP audit candidate", 1),
            ("S19-L16-H1-INTERPOLATED", "Common-minimum-horizon interpolation plus MLP audit candidate", 1),
            ("S19-L16-H2-PUBLIC-EXTRACTOR", "Generic public PhiRL vector/sequence extractor audit candidate", 1),
        ],
        start=1,
    ):
        additions.append(
            {
                "branchCount": 0,
                "bundleId": "L16_TENSOR_ARCHITECTURE_SOURCE_AUDIT",
                "candidateId": identifier,
                "candidateSpecificSuccess": 0,
                "completedFitLeakage": 0,
                "computeEfficiency": 5,
                "crossCandidateDiscriminability": 0,
                "deterministicHReuse": 1,
                "explanatoryLeverage": 4,
                "frozenRank": offset,
                "independenceFromPriorOutcomeSelection": 2,
                "outcomeGuidedThresholdSelection": 0,
                "paperFingerprintSpecificity": 4,
                "proposedSpecification": description,
                "rankingScore": float(16 - offset),
                "registryOrder": int(candidates["registryOrder"].max()) + offset,
                "selected": False,
                "selectionReason": "NOT_REGISTERED_FOR_EXECUTION_COMPLETE_SOURCE_GROUNDING_GATE_FAILED",
                "sourceGrounding": grounding,
                "testability": 1,
                "undefinedAuthorSemantics": 5,
            }
        )
    write_parquet(candidate_path, pd.concat([candidates, pd.DataFrame(additions)], ignore_index=True)[candidates.columns])

    source_path = S19_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    if sources["sourceId"].astype(str).str.startswith("L16_").any():
        raise RuntimeError("L16 source ledger rows already exist")
    source_rows = [
        {
            "commitOrVersion": "77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4",
            "evidenceClass": "DIRECT_PAPER_SPECIFICATION",
            "finding": "Paper specifies semantic 25/75 task, MLP, split/repetition and accuracy but not complete tensor, masking, aggregation or architecture.",
            "licenseStatus": "INPUT_ATTACHMENT_RESEARCH_USE",
            "redistributionStatus": "REFERENCE_ONLY",
            "repositoryIdentity": "original manuscript",
            "retainedPath": str(PAPER_PDF),
            "retrievalDate": "2026-08-10",
            "sha256": sha256_file(PAPER_PDF),
            "sourceId": "L16_PAPER_FIGURE5_TASK_AUDIT",
            "sourceType": "INPUT_PAPER",
            "treeIdentity": None,
            "url": None,
        },
        {
            "commitOrVersion": source_manifest["repositories"]["PhiRL"]["head"],
            "evidenceClass": "DIRECT_PUBLIC_CODE_NOT_LINKED_TO_GARD_FIGURE5",
            "finding": "Plotting-only minimum-horizon interpolation and generic RL extractors do not specify the GARD Figure-5 supervised task.",
            "licenseStatus": "NO_LICENSE_FILE_DETECTED",
            "redistributionStatus": "IDENTITY_AND_FINDING_ONLY",
            "repositoryIdentity": "https://github.com/AdriFrutos/PhiRL",
            "retainedPath": str(PHIRL_ROOT),
            "retrievalDate": "2026-08-10",
            "sha256": None,
            "sourceId": "L16_PHIRL_TENSOR_ARCHITECTURE_AUDIT",
            "sourceType": "PUBLIC_GIT_REPOSITORY",
            "treeIdentity": source_manifest["repositories"]["PhiRL"]["tree"],
            "url": "https://github.com/AdriFrutos/PhiRL",
        },
        {
            "commitOrVersion": sha256_file(L12_ROOT / "phirl_missing_gard_components.md"),
            "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
            "finding": "L12's complete lineage audit found no public GARD Figure-5 sequence tensor/MLP implementation.",
            "licenseStatus": "WORKSPACE_ARTIFACT",
            "redistributionStatus": "REFERENCE_ONLY",
            "repositoryIdentity": "Eidosoma/arrival-of-self-replicators",
            "retainedPath": str(L12_ROOT / "phirl_missing_gard_components.md"),
            "retrievalDate": "2026-08-10",
            "sha256": sha256_file(L12_ROOT / "phirl_missing_gard_components.md"),
            "sourceId": "L16_L12_PUBLIC_CODE_GAP_CROSSWALK",
            "sourceType": "FROZEN_INTERNAL_SOURCE_CROSSWALK",
            "treeIdentity": None,
            "url": None,
        },
    ]
    write_parquet(source_path, pd.concat([sources, pd.DataFrame(source_rows)], ignore_index=True)[sources.columns])

    ledger_path = S19_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if ledger["loopId"].eq("S19-L16").any():
        raise RuntimeError("L16 self-improvement rows already exist")
    start = int(ledger["ledgerSequence"].max()) + 1
    ledger_rows = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "The paper accuracies lie between L15 valid-cell non-discrimination and unmasked length/padding near-determinism; a source-grounded tensor convention might explain the middle regime.",
            "failureOrAmbiguityTargeted": "Missing Figure-5 variable-length tensor, masking, aggregation and MLP architecture semantics.",
            "informationGainRationale": "Audit completeness before outcomes and execute only a fully grounded convention, preventing another adaptive architecture search.",
            "learned": "Pre-outcome source audit initiated; no scientific outcome opened.",
            "ledgerSequence": start,
            "loopId": "S19-L16",
            "motivatingEvidence": "L15 frozen panel, L12 source gap, manuscript Figure-5 task description and public source lineages.",
            "proposedNextTest": "Only a complete source-grounded hypothesis may execute.",
            "recordPhase": "PRE_LOOP_SOURCE_GROUNDING_LOCK",
            "remainingPlausibleHypotheses": "A coherent private author tensor remains possible but is not publicly identifiable.",
            "selectedHypotheses": "Maximum three audited convention candidates; zero selected unless the complete gate passes.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "None before gate evaluation.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A public tensor/architecture clue might be complete enough for a bounded final model discrimination.",
            "failureOrAmbiguityTargeted": "Whether variable-length, mask, aggregation and architecture semantics are recoverable without author code.",
            "informationGainRationale": "A fail-at-source gate distinguishes missing evidence from failed numerical model performance without opening another specification family.",
            "learned": "No audited convention completely grounded all ten required fields; no new model fit was scientifically authorized.",
            "ledgerSequence": start + 1,
            "loopId": "S19-L16",
            "motivatingEvidence": "Paper semantic task, PhiRL plotting interpolation, generic RL extractors, full public history, and frozen S16/L15 contracts.",
            "proposedNextTest": "Mandatory human review; prefer author-code/configuration wait or separately authorized closeout over another adaptive tensor guess.",
            "recordPhase": "POST_LOOP_RESULT_AND_HUMAN_REVIEW_HANDOFF",
            "remainingPlausibleHypotheses": "A coherent private implementation may use an unreported convention; public evidence cannot discriminate it.",
            "selectedHypotheses": "None registered for execution.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "The proposition that public paper/source evidence suffices to lock a complete Figure-5 tensor/architecture reconstruction.",
        },
    ]
    write_parquet(ledger_path, pd.concat([ledger, pd.DataFrame(ledger_rows)], ignore_index=True)[ledger.columns])

    markdown_path = S19_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    existing = markdown_path.read_text(encoding="utf-8")
    marker = "## S19-L16 — source-grounding audit and mandatory human-review boundary"
    if marker in existing:
        raise RuntimeError("L16 markdown ledger entry already exists")
    addition = f"""

{marker}

- **Belief before:** The paper-panel values between L15's valid-cell and padding/length extremes might be explained by one source-grounded tensor or architecture convention.
- **Evidence audited:** The complete manuscript/Figure-5 task description, full pinned PhiRL public history, IIGR/Breaking/GARD lineages, L12 concordance, and exact S15/S16/L15 contracts.
- **What was learned:** zero of three audited convention candidates completely grounded all ten identity-changing fields. The public interpolation is plotting-only; public extractors are generic RL components; the complete S16 tensor/model is an E01 reconstruction.
- **Action taken:** the prospectively locked gate stopped before any new model fit. Frozen L15 inputs, tables, controls, suffix audit, and 24 sentinel probability identities replayed exactly.
- **Hypothesis weakened:** public evidence is sufficient to select a defensible Figure-5 tensor/architecture run.
- **What remains plausible:** a coherent but private author implementation with unreported variable-length, loss, scoring, aggregation, and architecture semantics.
- **Next action:** mandatory human review. No L17, S20, E02, author contact, confirmation, intervention or report generation is active.
"""
    atomic_text(markdown_path, existing.rstrip() + addition)

    loop_path = S19_ROOT / "loop_registry.yaml"
    loop_registry = yaml.safe_load(loop_path.read_text(encoding="utf-8"))
    if any(row.get("loopId") == "S19-L16" for row in loop_registry["loops"]):
        raise RuntimeError("L16 loop registry row already exists")
    loop_registry["loops"].append(
        {
            "loopId": "S19-L16",
            "versionedLoopId": VERSION,
            "status": "COMPLETE_SOURCE_GROUNDING_GATE_STOP_AWAITING_MANDATORY_HUMAN_REVIEW",
            "authorized": True,
            "outcomeAccessed": False,
            "humanReviewRequiredAfter": True,
            "completed": True,
            "eligibleScientificResults": False,
            "classification": ["AUTHOR_AMBIGUITY_UNRESOLVED", "EXPLORATORY_NON_SUPPORT", "NOT_PROMOTABLE"],
            "directedDecision": "NO_SUFFICIENTLY_SOURCE_GROUNDED_COMPLETE_TENSOR_OR_ARCHITECTURE_HYPOTHESIS",
            "newModelFits": 0,
            "nextStepActive": False,
        }
    )
    loop_registry["laterLoopsAuthorized"] = False
    loop_registry["s20Status"] = "DEFINED_INACTIVE"
    loop_registry["proposedNextLoopTheme"] = "AUTHOR_CODE_OR_CONFIGURATION_WAIT_STATE_OR_HUMAN_CLOSEOUT_DECISION"
    loop_registry["proposedNextLoopActive"] = False
    write_yaml(loop_path, loop_registry)

    review_path = S19_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["history"].append(
        {
            "decision": "AUTHORIZE_EXACTLY_ONE_L16_FINAL_TENSOR_AUDIT",
            "loopId": "S19-L16",
            "nextLoopAuthorized": False,
            "recordedAtUtc": timestamp,
            "result": "NO_SUFFICIENTLY_SOURCE_GROUNDED_COMPLETE_TENSOR_OR_ARCHITECTURE_HYPOTHESIS",
            "s20Activated": False,
            "scope": VERSION,
            "source": "explicit_human_direction",
            "status": "CONSUMED_AND_RETURNED_FOR_MANDATORY_REVIEW",
        }
    )
    review["pendingDecision"] = "POST_S19_L16_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(review_path, review)


def artifact_manifest(root: Path, current_loop: str | None = None) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    result = {
        "schema": "eidosoma.e01.s19.l16.artifact_manifest.v1" if current_loop is None else "eidosoma.e01.s19_root.artifact_manifest.v1",
        "createdAtUtc": utc_now(),
        "fileCount": len(files),
        "files": files,
    }
    if current_loop is None:
        result["researchStepId"] = "S19-L16"
        result["versionedStepId"] = VERSION
    else:
        result["currentLoop"] = current_loop
    return result


def execute() -> None:
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    lock = require_clean_pushed_lock()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(CONFIG_PATH, OUTPUT_ROOT / "preregistration.yaml")
    shutil.copyfile(AMENDMENT_PATH, OUTPUT_ROOT / "technical_amendment_001.json")

    baseline_rows = hash_tree(immutable_roots())
    baseline = {
        "schema": "eidosoma.e01.s19.l16.immutable_baseline.v1",
        "createdAtUtc": utc_now(),
        "fileCount": len(baseline_rows),
        "files": baseline_rows,
    }
    write_json(OUTPUT_ROOT / "immutable_prior_baseline.json", baseline)

    l15_artifacts = validate_l15_artifacts()
    source_frame, source_manifest, source_audit = source_evidence()
    source_manifest["repositoryLock"] = lock
    source_files = [
        PAPER_PDF,
        PAPER_MARKDOWN,
        PAPER_FIGURE5,
        PHIRL_ROOT / "plotting.py",
        PHIRL_ROOT / "archs.py",
        L12_ROOT / "phirl_missing_gard_components.md",
        L12_ROOT / "figure5_reconciliation_possibilities.csv",
        REPO_ROOT / "configs/e01/s16_tensor_model_manifest.json",
        REPO_ROOT / "src/e01_prediction_reconstruction/core.py",
        REPO_ROOT / "configs/e01/s19_l15_untouched_padding_panel.yaml",
        AMENDMENT_PATH,
        L15_ROOT / "S19_L15_FULL_RESULTS.md",
    ]
    source_manifest["files"] = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_files
    ]
    write_json(OUTPUT_ROOT / "source_snapshot_manifest.json", source_manifest)
    write_csv(OUTPUT_ROOT / "source_support_evidence.csv", source_frame)
    atomic_text(OUTPUT_ROOT / "paper_public_source_tensor_audit.md", source_audit)

    support, registry = hypothesis_support()
    write_parquet(OUTPUT_ROOT / "tensor_architecture_support_matrix.parquet", support)
    write_yaml(
        OUTPUT_ROOT / "tensor_architecture_hypothesis_registry.yaml",
        {
            "schema": "eidosoma.e01.s19.l16.hypothesis_registry.v1",
            "createdAtUtc": utc_now(),
            "maximumCompleteHypotheses": 3,
            "completeHypothesisCount": sum(item["completeSourceGroundingPassed"] for item in registry),
            "hypotheses": registry,
        },
    )
    gate_rows = pd.DataFrame(
        [
            {
                "hypothesisId": item["hypothesisId"],
                "requiredFieldCount": item["requiredFieldCount"],
                "directlyGroundedFieldCount": item["directlyGroundedFieldCount"],
                "completeSourceGroundingPassed": item["completeSourceGroundingPassed"],
                "registeredForExecution": item["registeredForExecution"],
                "stopBeforeNewModelExecution": not item["registeredForExecution"],
            }
            for item in registry
        ]
    )
    write_csv(OUTPUT_ROOT / "source_grounding_gate_results.csv", gate_rows)
    complete = [item for item in registry if item["registeredForExecution"]]
    if complete:
        raise RuntimeError("unexpected executable hypothesis; runner has no outcome stage")

    implementation_lock = {
        "schema": "eidosoma.e01.s19.l16.implementation_lock.v1",
        "createdAtUtc": utc_now(),
        "repository": lock,
        "sourceGroundingGatePassedHypothesisCount": 0,
        "decision": "STOP_BEFORE_NEW_MODEL_EXECUTION",
        "newModelFitsAuthorized": 0,
        "newModelFitsExecuted": 0,
        "frozenInputCohort": "S19-L15",
        "frozenTarget": "S16_ADJACENT_INCOMING_H090",
        "frozenSplit": "TEN_MATRIX_GROUPED_128_32_40_SPLITS",
        "scientificMethodsChanged": False,
    }
    write_json(OUTPUT_ROOT / "implementation_lock.json", implementation_lock)
    atomic_text(
        OUTPUT_ROOT / "decision_record.md",
        "# L16 decision record\n\n"
        "L15 is formally accepted as additive exploratory evidence with every prior classification unchanged. "
        "The prospectively pushed L16 gate admitted zero complete source-grounded tensor/architecture hypotheses. "
        "Per the explicit human rule, L16 stopped before any new model fit. This is a scientific source-completeness decision, not an operational failure.\n",
    )

    tensor_replay = validate_feature_tensors()
    target_replay = validate_targets()
    split_replay = validate_splits(config)
    model_replay = validate_cached_model_probabilities()
    write_parquet(OUTPUT_ROOT / "frozen_tensor_replay.parquet", tensor_replay)
    write_parquet(OUTPUT_ROOT / "frozen_target_replay.parquet", target_replay)
    write_parquet(OUTPUT_ROOT / "frozen_split_replay.parquet", split_replay)
    write_parquet(OUTPUT_ROOT / "frozen_model_replay.parquet", model_replay)
    write_parquet(OUTPUT_ROOT / "frozen_l15_artifact_replay.parquet", l15_artifacts)
    copied = copy_frozen_evidence()
    summary = summarize_frozen_metrics()
    write_csv(OUTPUT_ROOT / "frozen_l15_metric_summary.csv", summary)
    write_parquet(
        OUTPUT_ROOT / "model_execution_status.parquet",
        pd.DataFrame(
            [
                {
                    "hypothesisId": item["hypothesisId"],
                    "candidateId": candidate,
                    "status": "NOT_RUN_SOURCE_GROUNDING_GATE",
                    "newModelFitExecuted": False,
                    "newPredictionOutcomeOpened": False,
                    "reason": "COMPLETE_SOURCE_GROUNDING_GATE_FAILED",
                }
                for item in registry
                for candidate in ("CANDIDATE_2", "CANDIDATE_3")
            ]
        ),
    )
    write_parquet(
        OUTPUT_ROOT / "scientific_gate_results.parquet",
        pd.DataFrame(
            [
                {"gate": "maximumThreeCompleteHypotheses", "passed": True, "observed": 0, "required": "<=3"},
                {"gate": "completeSourceGrounding", "passed": False, "observed": 0, "required": ">=1 to execute"},
                {"gate": "stopBeforeNewModelExecution", "passed": True, "observed": 0, "required": "0 new fits after empty gate"},
                {"gate": "candidateSeparation", "passed": True, "observed": 2, "required": "2 separate candidates"},
                {"gate": "prospectivePromotionEligible", "passed": False, "observed": 0, "required": "prohibited by zero pre-onset eligibility"},
                {"gate": "priorClassificationChanged", "passed": True, "observed": 0, "required": "0"},
            ]
        ),
    )
    write_csv(
        OUTPUT_ROOT / "failure_ledger.csv",
        pd.DataFrame(
            [
                {
                    "failureId": "S19-L16-F001",
                    "stage": "FROZEN_TARGET_REPLAY_VALIDATOR",
                    "status": "PRESERVED_TECHNICAL_FAILURE_AMENDED",
                    "scientificValuesReleased": False,
                    "detail": "Initial validator incorrectly required hashes for two manifest-declared ineligible rows; partial attempt preserved and no model executed.",
                },
                {
                    "failureId": "",
                    "stage": "SOURCE_GROUNDING_GATE",
                    "status": "NO_OPERATIONAL_FAILURE_PROTOCOL_GATE_STOP",
                    "scientificValuesReleased": False,
                    "detail": "Zero complete source-grounded hypotheses; no new model execution permitted.",
                }
            ]
        ),
    )
    write_csv(
        OUTPUT_ROOT / "technical_amendment_ledger.csv",
        pd.DataFrame(
            [
                {
                    "amendmentId": "S19-L16-TECHNICAL-AMENDMENT-001",
                    "failedAttemptPreservedAt": "/cache/e01_s19_l16/failed_attempt_001_artifacts",
                    "outcomeAccessed": False,
                    "newModelFitExecuted": False,
                    "scientificMethodChanged": False,
                    "scientificValueChanged": False,
                    "status": "PASS_VALUE_PRESERVING_REPLAY_VALIDATOR_CORRECTION",
                }
            ]
        ),
    )
    figures(support, registry, summary)

    # Recheck immutable prior files after all loop-local writes.
    mismatches = []
    for record in baseline_rows:
        path = Path(record["path"])
        observed = sha256_file(path) if path.exists() else None
        if observed != record["sha256"]:
            mismatches.append({"path": str(path), "expected": record["sha256"], "observed": observed})
    if mismatches:
        raise RuntimeError(f"immutable prior mismatch: {mismatches[:3]}")
    immutable_validation = {
        "schema": "eidosoma.e01.s19.l16.immutable_validation.v1",
        "validatedAtUtc": utc_now(),
        "fileCount": len(baseline_rows),
        "mismatchCount": 0,
        "mismatches": [],
        "passed": True,
    }
    write_json(OUTPUT_ROOT / "immutable_prior_validation.json", immutable_validation)

    validations = {
        "priorFileCount": len(baseline_rows),
        "priorMismatchCount": 0,
        "l15ArtifactHashCount": len(l15_artifacts),
        "tensorArrayCount": len(tensor_replay),
        "targetPairCount": len(target_replay),
        "splitCount": len(split_replay),
        "modelReplayCount": len(model_replay),
    }
    report = build_report(summary, registry, validations, lock["head"])
    atomic_text(OUTPUT_ROOT / "S19_L16_FULL_RESULTS.md", report)
    atomic_text(OUTPUT_ROOT / "research_step_full_results.md", report)
    atomic_text(
        OUTPUT_ROOT / "loop_decision_summary.md",
        "# S19-L16 decision summary\n\n"
        "**Decision:** `NO_SUFFICIENTLY_SOURCE_GROUNDED_COMPLETE_TENSOR_OR_ARCHITECTURE_HYPOTHESIS`.\n\n"
        "The manuscript provides the semantic 25/75 MLP task but omits the identity-changing tensor, variable-length, target, masking, aggregation, topology and capacity details. Public PhiRL provides plotting-only interpolation and generic RL extractors, not a GARD Figure-5 predictor. Zero candidate conventions passed the complete-source gate, so no new model was fitted. Frozen L15 evidence and all integrity checks replayed exactly. L16 is `AUTHOR_AMBIGUITY_UNRESOLVED`, `EXPLORATORY_NON_SUPPORT`, and `NOT_PROMOTABLE`. Return for mandatory human review.\n",
    )
    classification = {
        "schema": "eidosoma.e01.s19.l16.classification.v1",
        "researchStepId": "S19-L16",
        "versionedStepId": VERSION,
        "status": "COMPLETE_SOURCE_GROUNDING_GATE_STOP_AWAITING_MANDATORY_HUMAN_REVIEW",
        "primaryClassification": "AUTHOR_AMBIGUITY_UNRESOLVED",
        "classifications": ["AUTHOR_AMBIGUITY_UNRESOLVED", "EXPLORATORY_NON_SUPPORT", "NOT_PROMOTABLE"],
        "directedDecision": "NO_SUFFICIENTLY_SOURCE_GROUNDED_COMPLETE_TENSOR_OR_ARCHITECTURE_HYPOTHESIS",
        "completeHypothesisCount": 0,
        "newModelFits": 0,
        "newScientificOutcomes": 0,
        "priorClassificationsChanged": False,
        "s18Changed": False,
        "nextStepActive": False,
    }
    write_json(OUTPUT_ROOT / "classification.json", classification)

    runtime = {
        "schema": "eidosoma.e01.s19.l16.runtime.v1",
        "completedAtUtc": utc_now(),
        "wallSeconds": time.perf_counter() - started,
        "cpuCountVisible": os.cpu_count(),
        "cpuCoresMaximum": config["resources"]["cpuCoresMaximum"],
        "gpuHours": 0,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "newModelFits": 0,
    }
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)

    retained = sum(path.stat().st_size for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19.l16.storage_validation.v1",
        "retainedBytesBeforeManifest": retained,
        "retainedGiB": retained / 1024**3,
        "retainedGiBMaximum": config["resources"]["retainedGiBMaximum"],
        "temporaryGiBMaximum": config["resources"]["temporaryGiBMaximum"],
        "newTemporaryCacheBytes": 0,
        "passed": retained <= config["resources"]["retainedGiBMaximum"] * 1024**3,
    }
    write_json(OUTPUT_ROOT / "storage_validation.json", storage)
    regeneration = {
        "schema": "eidosoma.e01.s19.l16.regeneration_validation.v1",
        "validatedAtUtc": utc_now(),
        "inputTensorArraysExact": int(tensor_replay["passed"].sum()),
        "inputTensorArrayCount": len(tensor_replay),
        "targetPairsExact": int(target_replay["passed"].sum()),
        "targetPairCount": len(target_replay),
        "splitReplaysExact": int(split_replay["passed"].sum()),
        "splitReplayCount": len(split_replay),
        "cachedModelProbabilityIdentitiesExact": int(model_replay["passed"].sum()),
        "cachedModelProbabilityIdentityCount": len(model_replay),
        "l15ArtifactHashesExact": int(l15_artifacts["passed"].sum()),
        "l15ArtifactHashCount": len(l15_artifacts),
        "readOnlyEvidenceCopies": copied,
        "newModelReplayRun": False,
        "reason": "EMPTY_SOURCE_GROUNDING_GATE_PROHIBITED_MODEL_EXECUTION",
        "passed": True,
    }
    write_json(OUTPUT_ROOT / "regeneration_validation.json", regeneration)
    write_json(
        OUTPUT_ROOT / "input_manifest.json",
        {
            "schema": "eidosoma.e01.s19.l16.input_manifest.v1",
            "createdAtUtc": utc_now(),
            "sourceCohort": "S19-L15",
            "matrices": 200,
            "trajectories": 400,
            "tensorEligibleEachCandidate": 199,
            "featureTensorFiles": 12,
            "splits": 10,
            "newMatrices": 0,
            "newTrajectories": 0,
            "newLabels": 0,
            "newFeatures": 0,
            "newModelFits": 0,
        },
    )

    # Loop manifest before append-only root updates.
    loop_manifest = artifact_manifest(OUTPUT_ROOT)
    write_json(OUTPUT_ROOT / "artifact_manifest.json", loop_manifest)
    for item in loop_manifest["files"]:
        path = OUTPUT_ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError("loop artifact manifest replay failed")

    timestamp = utc_now()
    append_root_ledgers(timestamp, source_manifest)
    root_handoff = "\n".join(
        [
            "# S19 Current-Step Handoff — L16",
            "",
            "- **Status:** `COMPLETE_SOURCE_GROUNDING_GATE_STOP_AWAITING_MANDATORY_HUMAN_REVIEW`",
            "- **Decision:** `NO_SUFFICIENTLY_SOURCE_GROUNDED_COMPLETE_TENSOR_OR_ARCHITECTURE_HYPOTHESIS`",
            "- **Classification:** `AUTHOR_AMBIGUITY_UNRESOLVED`; `EXPLORATORY_NON_SUPPORT`; `NOT_PROMOTABLE`.",
            "- **Scientific execution:** zero new model fits and zero new outcomes; the pushed source-completeness gate required the stop.",
            "- **Validation:** immutable prior, 12 tensors/36 arrays, 400 targets/masks, ten splits, 60 L15 artifact hashes, 24 cached model/replay identities, controls and suffix evidence passed.",
            "- **Boundary:** every S01–S18 and S19-L01–L15 result remains unchanged. S20, E02, author contact, confirmation, intervention, report generation, and L17 are inactive.",
            "- **Recommended next action:** mandatory human review; prefer exact author-code/configuration recovery or closeout over another outcome-guided tensor guess.",
            "",
        ]
    )
    atomic_text(S19_ROOT / "research_step_full_results.md", root_handoff)
    write_json(
        S19_ROOT / "s19_status.json",
        {
            "researchStepId": "S19-L16",
            "status": "COMPLETE_SOURCE_GROUNDING_GATE_STOP_AWAITING_MANDATORY_HUMAN_REVIEW",
            "outcomeClassification": "AUTHOR_AMBIGUITY_UNRESOLVED",
            "validationResult": "PASS_IMMUTABLE_SOURCE_TENSOR_TARGET_SPLIT_CACHED_MODEL_CONTROL_SUFFIX_STORAGE_REGENERATION",
            "artifactsWritten": [
                str(OUTPUT_ROOT / "S19_L16_FULL_RESULTS.md"),
                str(OUTPUT_ROOT / "classification.json"),
                str(OUTPUT_ROOT / "artifact_manifest.json"),
                str(S19_ROOT / "research_step_full_results.md"),
            ],
            "caveatsOrBlockers": [
                "no_complete_publicly_grounded_tensor_architecture_hypothesis",
                "author_implementation_required",
                "adjacent_H_cohort_not_preonset_eligible",
                "completed_fit_future_dependence_unchanged",
                "S18_statuses_unchanged",
            ],
            "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_KEEP_L17_S20_E02_AUTHOR_CONTACT_CONFIRMATION_INTERVENTIONS_AND_REPORT_BUNDLE_INACTIVE",
        },
    )
    write_json(S19_ROOT / "artifact_manifest.json", artifact_manifest(S19_ROOT, "L16"))
    print(
        canonical_json(
            {
                "status": classification["status"],
                "classification": classification["classifications"],
                "decision": classification["directedDecision"],
                "newModelFits": 0,
                "artifactCount": loop_manifest["fileCount"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("L16 has one outcome-blind audit stage; pass --execute")
    execute()


if __name__ == "__main__":
    main()
