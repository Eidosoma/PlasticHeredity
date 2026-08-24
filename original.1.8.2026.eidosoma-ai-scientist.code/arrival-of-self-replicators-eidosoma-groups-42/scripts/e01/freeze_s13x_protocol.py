#!/usr/bin/env python3
"""Freeze the S13X adaptive protocol, gap ranking, and immutable input baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from e01_creative_directional_search.core import (
    EVIDENCE_CLASS,
    RESEARCH_STEP_ID,
    VERSION,
    label_specs,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps" / "S13X"
CONFIG = REPO / "configs" / "e01" / "s13x_adaptive_protocol.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot infer empty CSV schema for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def prior_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = ARTIFACTS / "research_steps"
    for path in sorted(root.glob("S*/**/*")):
        if not path.is_file() or path.is_relative_to(STEP_ROOT):
            continue
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def raw_inputs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"):
        for matrix_index in range(100):
            path = Path(
                f"/cache/e01_s13/raw_trajectories/{candidate}/M{matrix_index:03d}.pickle"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "inputKind": "FROZEN_S13_TRAJECTORY",
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for input_kind, path in (
        (
            "S13RRR_FULL_SOURCE_VALUES",
            ARTIFACTS / "research_steps" / "S13RRR" / "full_source_values.parquet",
        ),
        (
            "S13RRR_LABEL_VALUES",
            ARTIFACTS / "research_steps" / "S13RRR" / "label_values.parquet",
        ),
        (
            "S12B_SAFE_LATTICE",
            ARTIFACTS / "research_steps" / "S12B" / "safe_phi_lattice.json",
        ),
        (
            "PAPER_MARKDOWN",
            Path(
                "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
            ),
        ),
        (
            "AMBIGUITY_LEDGER",
            ARTIFACTS
            / "E01_forensic_replication_bundle"
            / "ledgers"
            / "ambiguity_ledger.csv",
        ),
        (
            "CLAIM_LEDGER",
            ARTIFACTS
            / "E01_forensic_replication_bundle"
            / "ledgers"
            / "claim_ledger.csv",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "inputKind": input_kind,
                "candidateId": None,
                "matrixIndex": None,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def gap_inventory() -> list[dict[str, Any]]:
    return [
        {
            "rank": 1,
            "gapId": "GAP-LABEL-OBSERVATION-SCOPE",
            "layer": "replicator_labels",
            "leverage": "VERY_HIGH",
            "evidenceTier": "PAPER_AND_SOURCE_CONFLICT",
            "priorFinding": "S13 historical post-fission adjacent-H labels occupied about 36% of generations, far below Table 1's 88% molecular-step probability.",
            "paperClue": "The paper says every time step is compared with the most recurring composition and reports molecular-step persistence/probability.",
            "S13XAction": "Compare post-fission recurring-composition, Euclidean, and molecular-clock adjacent-state families; retain full threshold ledger.",
        },
        {
            "rank": 2,
            "gapId": "GAP-LEVEL-VERSUS-CHANGE",
            "layer": "association",
            "leverage": "VERY_HIGH",
            "evidenceTier": "DIRECT_PAPER_DISCREPANCY",
            "priorFinding": "S13 tested source metric levels; both confirmed candidates were near zero or negative.",
            "paperClue": "Results prose says correlation with Phi-r, while Figure 3 explicitly says changes in Phi-r.",
            "S13XAction": "Compare levels, backward/forward changes, positive/absolute changes, trailing summaries, and generation summaries.",
        },
        {
            "rank": 3,
            "gapId": "GAP-TEMPORAL-LABEL-PLACEMENT",
            "layer": "alignment",
            "leverage": "HIGH",
            "evidenceTier": "UNRESOLVED_IMPLEMENTATION",
            "priorFinding": "A generation endpoint label was propagated across its own growth interval in S13.",
            "paperClue": "The paper does not state whether endpoint classification labels the preceding growth, following daughter lineage, or exact state.",
            "S13XAction": "Test same/next/previous molecular state and next/previous generation alignments.",
        },
        {
            "rank": 4,
            "gapId": "GAP-METRIC-ATOM-IDENTITY",
            "layer": "information_metric",
            "leverage": "HIGH",
            "evidenceTier": "SOURCE_AND_EQUATION_CONFLICT",
            "priorFinding": "S12C local_phi_r and S12D synergy-plus-downward emergence were both non-supportive under frozen labels.",
            "paperClue": "The displayed equation, causal-emergence prose, and public source distinguish integrated, emergence, synergy, and downward-causation quantities.",
            "S13XAction": "Compare all four already materialized source scalars without hiding unfavorable atoms.",
        },
        {
            "rank": 5,
            "gapId": "GAP-PREPROCESSING-AFTER-CLR",
            "layer": "preprocessing",
            "leverage": "HIGH",
            "evidenceTier": "PAPER_VERSUS_PUBLIC_SOURCE",
            "priorFinding": "S13 used public IIGR/PhiRL preprocessing after the paper's CLR substrate.",
            "paperClue": "The paper reports closure/CLR/drop-last only; public GRN code additionally z-scores, regresses global signal, and residualizes AR(1).",
            "S13XAction": "If the existing-output screen is insufficient, refit CLR-direct and z-scored variants under explicit partition alternatives.",
        },
        {
            "rank": 6,
            "gapId": "GAP-MIB-PARTITION-IDENTITY",
            "layer": "partition",
            "leverage": "HIGH",
            "evidenceTier": "PAPER_VERSUS_PUBLIC_SOURCE",
            "priorFinding": "Public source uses an unnormalized graph Fiedler split, while the paper calls the partition a minimum-information bipartition.",
            "paperClue": "No normalization, objective, balance rule, or search method is reported.",
            "S13XAction": "Compare source unnormalized, normalized-laplacian, and balanced spectral partitions in the focused refit phase.",
        },
        {
            "rank": 7,
            "gapId": "GAP-LOCAL-FIT-WINDOW",
            "layer": "estimation",
            "leverage": "MEDIUM_HIGH",
            "evidenceTier": "UNRESOLVED_AND_PRIOR_FAILURE",
            "priorFinding": "Completed-fit public-source values are future-dependent; strict and repaired fixed-window branches failed earlier validation.",
            "paperClue": "Figures appear local but no window or fit scope is reported.",
            "S13XAction": "Use full-fit outputs as retrospective exploration and trailing summaries; do not relabel them prospective.",
        },
        {
            "rank": 8,
            "gapId": "GAP-GARD-SEMANTICS",
            "layer": "simulation",
            "leverage": "MEDIUM",
            "evidenceTier": "UPSTREAM_NONIDENTIFIABILITY",
            "priorFinding": "S12FR found three paper-compatible time bases; S13 held out candidates 2 and 3 and found near-zero source associations.",
            "paperClue": "Exposure, daughter selection, overshoot, and observation boundaries remain unpublished.",
            "S13XAction": "Carry both confirmed candidates through the analytical search; revisit kinetics only if downstream choices cannot explain the discrepancy.",
        },
        {
            "rank": 9,
            "gapId": "GAP-SPIKE-DEFINITION",
            "layer": "spikes",
            "leverage": "MEDIUM",
            "evidenceTier": "UNRESOLVED_SCOPE",
            "priorFinding": "Prior source values were punctuated under per-run thresholds but aggregate trend tests differed from the paper.",
            "paperClue": "The phrase overall mean may mean run-wise, pooled, or aggregate-series moments.",
            "S13XAction": "Retain conventional and robust run-wise spikes and report trend/spike diagnostics for every top branch.",
        },
        {
            "rank": 10,
            "gapId": "GAP-INTERVENTION-SCORING",
            "layer": "intervention",
            "leverage": "CONDITIONAL",
            "evidenceTier": "UNDERDETERMINED",
            "priorFinding": "Strict S12 scoring suppressed 1,089/1,090 opportunities; S12E intervention semantics were never reached.",
            "paperClue": "The paper states raw max/min after fission but omits model-refit and candidate-scoring details.",
            "S13XAction": "Run only a small explicitly directional pilot if an association pipeline is plausibly paper-like.",
        },
    ]


def preliminary_attempts() -> list[dict[str, Any]]:
    return [
        {
            "attemptSequence": 0,
            "attemptId": "S13X-PRE-A000",
            "phase": "PRE_PROTOCOL_DISCOVERY",
            "choiceFamily": "REFRESH_AND_SCHEMA",
            "specification": "Read plans/paper/ledgers/S13RRR; inspect frozen source and raw schemas.",
            "evidenceTier": "DIRECT_INPUT_AUDIT",
            "outcome": "Identified label scope and level-versus-change as highest-leverage gaps.",
            "negativeResult": False,
            "selectionUse": "Created ranked gap inventory.",
        },
        {
            "attemptSequence": 1,
            "attemptId": "S13X-PRE-A001",
            "phase": "PRE_PROTOCOL_DISCOVERY",
            "choiceFamily": "POSTFISSION_RECURRING_COMPOSITION_LABEL",
            "specification": "Connected-component centroid and max-neighbor medoid, H=0.75..0.95, 200 frozen trajectories.",
            "evidenceTier": "PAPER_INFERRED_PLUS_SENSITIVITY",
            "outcome": "At H=0.90 median molecular occupancy was about 0.20-0.21 (centroid) and 0.19 (medoid), not Table-1-like.",
            "negativeResult": True,
            "selectionUse": "Retain in complete grid; do not treat most-recurring wording as resolved.",
        },
        {
            "attemptSequence": 2,
            "attemptId": "S13X-PRE-A002",
            "phase": "PRE_PROTOCOL_DISCOVERY",
            "choiceFamily": "MOLECULAR_ADJACENT_LABEL",
            "specification": "Incoming and historical-average consecutive-state cosine at H=0.80..0.999.",
            "evidenceTier": "SOURCE_TRANSPLANT_AND_TABLE1_DIRECTED",
            "outcome": "Average-H=0.975 on candidate 2 reproduced persistence 714.03±195.73 versus paper 716±198, but occupancy, consistency, and onset were not jointly exact.",
            "negativeResult": False,
            "selectionUse": "Elevated observation-scope/threshold interaction for systematic search; explicitly adaptive.",
        },
        {
            "attemptSequence": 3,
            "attemptId": "S13X-PRE-A003",
            "phase": "PRE_PROTOCOL_DISCOVERY",
            "choiceFamily": "IIGR_EMERGENCE_QUICK_ASSOCIATION",
            "specification": "Selected molecular-adjacent labels against existing IIGR full emergence, levels and first differences.",
            "evidenceTier": "ADAPTIVE_SCREEN",
            "outcome": "Best observed positive-run fraction in this narrow check was 0.59; promising direction but below the paper's 0.73 and not a selected result.",
            "negativeResult": False,
            "selectionUse": "Justified a complete metric/transform/alignment ledger rather than threshold cherry-picking.",
        },
    ]


def main() -> None:
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["versionedStepId"] != VERSION:
        raise RuntimeError("S13X protocol version mismatch")
    prior = prior_artifacts()
    inputs = raw_inputs()
    gaps = gap_inventory()
    attempts = preliminary_attempts()
    write_json(
        STEP_ROOT / "prior_artifact_baseline.json",
        {
            "schema": "eidosoma.e01.s13x_prior_artifact_baseline.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "files": prior,
            "fileCount": len(prior),
            "passed": bool(prior),
        },
    )
    write_json(
        STEP_ROOT / "input_manifest.json",
        {
            "schema": "eidosoma.e01.s13x_input_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "inputs": inputs,
            "inputCount": len(inputs),
            "passed": len(inputs) == 206,
        },
    )
    write_csv(STEP_ROOT / "ranked_gap_inventory.csv", gaps)
    write_csv(STEP_ROOT / "chronological_search_ledger.csv", attempts)
    write_json(
        STEP_ROOT / "label_registry.json",
        {
            "schema": "eidosoma.e01.s13x_label_registry.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "labels": [
                {
                    "labelId": item.label_id,
                    "family": item.family,
                    "threshold": item.threshold,
                    "evidenceTier": item.evidence_tier,
                    "rationale": item.rationale,
                }
                for item in label_specs()
            ],
        },
    )
    (STEP_ROOT / "adaptive_search_protocol.yaml").write_text(
        CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    head = git("rev-parse", "HEAD")
    validation = {
        "schema": "eidosoma.e01.s13x_protocol_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "repositoryHead": head,
        "repositoryBranch": git("branch", "--show-current"),
        "priorArtifactCount": len(prior),
        "inputCount": len(inputs),
        "rawTrajectoryCount": sum(
            item["inputKind"] == "FROZEN_S13_TRAJECTORY" for item in inputs
        ),
        "gapCount": len(gaps),
        "labelSpecificationCount": len(label_specs()),
        "preProtocolAttemptCount": len(attempts),
        "priorEvidenceMutable": False,
        "adaptiveOutcomeGuided": True,
        "passed": bool(
            prior
            and len(inputs) == 206
            and len(gaps) == 10
            and len(label_specs()) == 22
            and git("branch", "--show-current") == "eidosoma/groups/42"
        ),
    }
    write_json(STEP_ROOT / "protocol_validation.json", validation)
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
