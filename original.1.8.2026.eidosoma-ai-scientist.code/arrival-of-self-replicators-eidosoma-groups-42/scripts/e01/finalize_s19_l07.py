#!/usr/bin/env python3
"""Deterministically finalize the additive S19-L07 occupancy search."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.e01.run_s19_l07 import (
    LOOP_ROOT,
    REPO,
    ARTIFACT_ROOT,
    sha256_file,
    utc_now,
    validate_prior_baseline,
    write_csv,
    write_json,
    write_parquet,
)

ROUND_IDS = (
    "R01_BOUNDARY_CLOCK",
    "R02_THRESHOLD_TRANSCRIPTION",
    "R03_EXPOSURE_SIMULATOR",
    "R04_ADAPTIVE_EXPOSURE_REFINEMENT",
    "R05_EXPOSURE_LOCAL_BRACKETING",
    "R06_FRESH_SEED_VALIDATION",
)
VERSIONED_ID = "E01-S19-L07-OCCUPANCY-SETTING-SEARCH-v1.0.0"
COMPLETED_AT = "2026-08-09"


def _append_unique(frame: pd.DataFrame, additions: pd.DataFrame, key: str) -> pd.DataFrame:
    overlap = set(frame[key]).intersection(set(additions[key]))
    if overlap:
        raise RuntimeError(f"append-only ledger already contains {key} values {sorted(overlap)}")
    return pd.concat([frame, additions.reindex(columns=frame.columns)], ignore_index=True)


def _all_round_tables(suffix: str, reader: str = "parquet") -> pd.DataFrame:
    frames = []
    for round_id in ROUND_IDS:
        path = LOOP_ROOT / f"{round_id}_{suffix}"
        frames.append(pd.read_parquet(path) if reader == "parquet" else pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _selected_rows(
    results: pd.DataFrame, round_id: str, pair_id: str
) -> pd.DataFrame:
    return results.loc[
        results["roundId"].eq(round_id) & results["settingPairId"].eq(pair_id)
    ].sort_values("candidateId", kind="stable")


def _make_figure(
    results: pd.DataFrame, registry: pd.DataFrame, fingerprints: pd.DataFrame
) -> None:
    joined = results.merge(
        registry[
            [
                "settingId",
                "h",
                "daughterRule",
                "family",
                "alignment",
                "projection",
            ]
        ],
        on="settingId",
        how="left",
    )
    curve = joined.loc[
        joined["roundId"].isin(
            [
                "R03_EXPOSURE_SIMULATOR",
                "R04_ADAPTIVE_EXPOSURE_REFINEMENT",
                "R05_EXPOSURE_LOCAL_BRACKETING",
            ]
        )
        & joined["settingPairId"].str.endswith("C1_INCOMING_ALL_H900")
        & joined["candidateId"].isin(["FIRST_DAUGHTER", "RANDOM_NONEMPTY"])
    ].copy()
    fresh = joined.loc[
        joined["roundId"].eq("R06_FRESH_SEED_VALIDATION")
        & joined["settingPairId"].eq("L07-R06-H28750::C1_INCOMING_ALL_H900")
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"FIRST_DAUGHTER": "#1f77b4", "RANDOM_NONEMPTY": "#d95f02"}
    for candidate, group in curve.groupby("candidateId", sort=True):
        group = group.sort_values("h")
        axes[0].scatter(
            group["h"],
            group["meanOccupancy"],
            s=32,
            alpha=0.8,
            color=colors[candidate],
            label=candidate.replace("_", " ").title(),
        )
    for candidate, group in fresh.groupby("candidateId", sort=True):
        axes[0].scatter(
            group["h"],
            group["meanOccupancy"],
            s=150,
            marker="*",
            edgecolor="black",
            linewidth=0.7,
            color=colors[candidate],
            zorder=5,
        )
    axes[0].axhspan(0.85, 0.91, color="#cccccc", alpha=0.25, label="88% ± 3 points")
    axes[0].axhline(0.88, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Poisson exposure h", ylabel="Mean replicator occupancy", title="All molecular steps, strict incoming H > 0.9")
    axes[0].set_ylim(0.80, 1.005)
    axes[0].legend(fontsize=8, loc="lower left")

    comparisons = [
        ("Adjacent\nH>.9", "R01_BOUNDARY_CLOCK", "C1_INCOMING_ALL_H900"),
        ("Fission\nboundaries", "R01_BOUNDARY_CLOCK", "PARENT_DAUGHTER_BOUNDARY_ONLY_H900"),
        ("Boundary→\ninterval", "R01_BOUNDARY_CLOCK", "PARENT_DAUGHTER_OUTGOING_ELIGIBLE_H900"),
        ("C0 avg\nH>.97", "R02_THRESHOLD_TRANSCRIPTION", "C0_AVG_H9700"),
        ("Fresh h=2.875\nH>.9", "R06_FRESH_SEED_VALIDATION", "L07-R06-H28750::C1_INCOMING_ALL_H900"),
    ]
    x = np.arange(len(comparisons), dtype=float)
    width = 0.36
    for offset, candidate_index in ((-width / 2, 0), (width / 2, 1)):
        values = []
        labels = []
        for _, round_id, pair_id in comparisons:
            subset = _selected_rows(results, round_id, pair_id)
            row = subset.iloc[candidate_index]
            values.append(float(row["meanOccupancy"]))
            labels.append(str(row["candidateId"]))
        axes[1].bar(
            x + offset,
            values,
            width,
            color=("#1f77b4" if candidate_index == 0 else "#d95f02"),
            alpha=0.9,
            label=("Candidate/branch A" if candidate_index == 0 else "Candidate/branch B"),
        )
    axes[1].axhline(0.88, color="black", linestyle="--", linewidth=1)
    axes[1].axhspan(0.85, 0.91, color="#cccccc", alpha=0.25)
    axes[1].set_xticks(x, [item[0] for item in comparisons], fontsize=8)
    axes[1].set(ylabel="Mean replicator occupancy", title="Distinct routes to the paper-level occupancy")
    axes[1].set_ylim(0.80, 1.005)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("S19-L07 exploratory occupancy reconstruction (occupancy-only success target)")
    fig.savefig(LOOP_ROOT / "occupancy_search_summary.png", dpi=180)
    plt.close(fig)


def _append_root_ledgers() -> None:
    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    next_order = int(candidates["registryOrder"].max()) + 1
    additions = pd.DataFrame(
        [
            {
                "candidateId": "S19-L07-OCC-01",
                "bundleId": "L07_CLOCK_AND_BOUNDARY_DENOMINATOR",
                "selected": True,
                "sourceGrounding": 4,
                "paperFingerprintSpecificity": 5,
                "explanatoryLeverage": 5,
                "testability": 5,
                "crossCandidateDiscriminability": 5,
                "computeEfficiency": 5,
                "independenceFromPriorOutcomeSelection": 3,
                "outcomeGuidedThresholdSelection": 0,
                "deterministicHReuse": 0,
                "completedFitLeakage": 0,
                "candidateSpecificSuccess": 0,
                "undefinedAuthorSemantics": 2,
                "branchCount": 16,
                "proposedSpecification": "Molecular clocks and strict-H>0.9 parent-to-selected-daughter boundary denominators/projections",
                "selectionReason": "Paper describes inheritance across generations; exact occupancy object/denominator is missing",
                "rankingScore": 40.0,
                "frozenRank": 1,
                "registryOrder": next_order,
            },
            {
                "candidateId": "S19-L07-OCC-02",
                "bundleId": "L07_THRESHOLD_AND_TRANSCRIPTION",
                "selected": True,
                "sourceGrounding": 3,
                "paperFingerprintSpecificity": 4,
                "explanatoryLeverage": 4,
                "testability": 5,
                "crossCandidateDiscriminability": 5,
                "computeEfficiency": 5,
                "independenceFromPriorOutcomeSelection": 1,
                "outcomeGuidedThresholdSelection": 5,
                "deterministicHReuse": 0,
                "completedFitLeakage": 0,
                "candidateSpecificSuccess": 0,
                "undefinedAuthorSemantics": 3,
                "branchCount": 48,
                "proposedSpecification": "Frozen clocks with strict/average similarity and fixed threshold list",
                "selectionReason": "Explicitly tests preprint/configuration transcription while retaining every attempted threshold",
                "rankingScore": 19.0,
                "frozenRank": 3,
                "registryOrder": next_order + 1,
            },
            {
                "candidateId": "S19-L07-OCC-03",
                "bundleId": "L07_POISSON_EXPOSURE",
                "selected": True,
                "sourceGrounding": 3,
                "paperFingerprintSpecificity": 5,
                "explanatoryLeverage": 5,
                "testability": 5,
                "crossCandidateDiscriminability": 5,
                "computeEfficiency": 5,
                "independenceFromPriorOutcomeSelection": 2,
                "outcomeGuidedThresholdSelection": 0,
                "deterministicHReuse": 0,
                "completedFitLeakage": 0,
                "candidateSpecificSuccess": 0,
                "undefinedAuthorSemantics": 4,
                "branchCount": 100,
                "proposedSpecification": "Fixed-common Poisson exposure h with otherwise unchanged simulator and strict molecular H>0.9",
                "selectionReason": "Paper specifies Poisson updates but omits exposure duration",
                "rankingScore": 31.0,
                "frozenRank": 2,
                "registryOrder": next_order + 2,
            },
        ]
    )
    write_parquet(candidate_path, _append_unique(candidates, additions, "candidateId"))

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_manifest = json.loads((LOOP_ROOT / "source_snapshot_manifest.json").read_text())
    file_by_path = {item["path"]: item for item in source_manifest["sources"]}
    paper_path = "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
    parameter_path = "/cache/e01_s03/sources/gard-historical/tgs_parameters_v10.m"
    cluster_path = "/cache/e01_s03/sources/gard-historical/cluster_traces.m"
    source_additions = pd.DataFrame(
        [
            {
                "sourceId": "L07_TARGET_PAPER_OCCUPANCY_AND_GARD_DESCRIPTION",
                "sourceType": "ORIGINAL_PAPER_ATTACHMENT",
                "url": "https://arxiv.org/abs/2607.28250",
                "repositoryIdentity": None,
                "commitOrVersion": "2607.28250v1",
                "treeIdentity": None,
                "retrievalDate": COMPLETED_AT,
                "retainedPath": paper_path,
                "sha256": file_by_path[paper_path]["sha256"],
                "licenseStatus": "INPUT_ATTACHMENT_INTERNAL_USE",
                "evidenceClass": "DIRECT_PAPER_EVIDENCE",
                "finding": "Paper reports control occupancy 88+/-3%, describes recurring composition-space attractors across generations, and specifies Poisson updates without an exposure duration.",
                "redistributionStatus": "CITATION_AND_BOUNDED_SUMMARY_ONLY",
            },
            {
                "sourceId": "L07_HISTORICAL_GARD_H900_PARAMETERS",
                "sourceType": "PUBLIC_CODE_LINEAGE",
                "url": "https://github.com/ModelingOriginsofLife/GARD",
                "repositoryIdentity": "ModelingOriginsofLife/GARD",
                "commitOrVersion": "86dff6320d5ae91b4e831471079ff46749b14df9",
                "treeIdentity": None,
                "retrievalDate": COMPLETED_AT,
                "retainedPath": parameter_path,
                "sha256": file_by_path[parameter_path]["sha256"],
                "licenseStatus": "NO_LICENSE_FILE_FOUND_DO_NOT_REDISTRIBUTE_SOURCE",
                "evidenceClass": "DIRECT_PUBLIC_CODE_LINEAGE",
                "finding": "Historical GARD v10 sets the drift/non-drift H cutoff to 0.9 and represents one composition per generation.",
                "redistributionStatus": "IDENTITY_AND_FINDING_ONLY",
            },
            {
                "sourceId": "L07_HISTORICAL_GARD_CLUSTER_H095",
                "sourceType": "PUBLIC_CODE_LINEAGE",
                "url": "https://github.com/ModelingOriginsofLife/GARD",
                "repositoryIdentity": "ModelingOriginsofLife/GARD",
                "commitOrVersion": "86dff6320d5ae91b4e831471079ff46749b14df9",
                "treeIdentity": None,
                "retrievalDate": COMPLETED_AT,
                "retainedPath": cluster_path,
                "sha256": file_by_path[cluster_path]["sha256"],
                "licenseStatus": "NO_LICENSE_FILE_FOUND_DO_NOT_REDISTRIBUTE_SOURCE",
                "evidenceClass": "DIRECT_PUBLIC_CODE_LINEAGE",
                "finding": "A separate historical cluster helper uses 0.95, demonstrating version/metric-specific threshold plurality but not grounding 0.97 or identifying target-paper code.",
                "redistributionStatus": "IDENTITY_AND_FINDING_ONLY",
            },
            {
                "sourceId": "L07_FROZEN_S13Y_INPUT_CONTEXT",
                "sourceType": "FROZEN_INTERNAL_EVIDENCE",
                "url": None,
                "repositoryIdentity": None,
                "commitOrVersion": "E01-S13Y-PHIRL-SOURCE-SEMANTICS-v1.0.0",
                "treeIdentity": None,
                "retrievalDate": COMPLETED_AT,
                "retainedPath": "/artifacts/research_steps/S13Y/artifact_manifest.json",
                "sha256": sha256_file(Path("/artifacts/research_steps/S13Y/artifact_manifest.json")),
                "licenseStatus": "INTERNAL_GENERATED_EVIDENCE",
                "evidenceClass": "FROZEN_INTERNAL_EVIDENCE",
                "finding": "Frozen trajectories supply the 100 shared matrices, candidate identities, selected molecular clocks, exact H arrays, and H>0.9 comparator labels.",
                "redistributionStatus": "INTERNAL_OR_CITABLE_REFERENCE",
            },
        ]
    )
    write_parquet(source_path, _append_unique(sources, source_additions, "sourceId"))

    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max())
    timestamps = [
        json.loads((LOOP_ROOT / "R01_BOUNDARY_CLOCK_status.json").read_text())["startedAtUtc"],
        json.loads((LOOP_ROOT / "R06_FRESH_SEED_VALIDATION_status.json").read_text())["outcomeOpenedAtUtc"],
    ]
    ledger_additions = pd.DataFrame(
        [
            {
                "ledgerSequence": sequence + 1,
                "timestampUtc": timestamps[0],
                "loopId": "S19-L07",
                "recordPhase": "PRE_LOOP_OCCUPANCY_ONLY_WAIVER_AND_SELECTION",
                "beliefBeforeLoop": "The 98% versus 88% mismatch could arise from clock/denominator, threshold transcription, or an omitted simulator exposure rather than one uniquely identified label definition.",
                "motivatingEvidence": "Adjacent molecular H>0.9 exactly determines the frozen label and yields about 98%; prior H>0.97 sensitivity approached 88% without reproducing temporal structure.",
                "failureOrAmbiguityTargeted": "Identify reproducible settings that approach paper occupancy while retaining all attempts and without claiming author identity.",
                "selectedHypotheses": "Clock/boundary object, fixed threshold/transcription family, Poisson exposure sweep, adaptive exposure bracket, and fresh-seed validation.",
                "learned": None,
                "weakenedHypotheses": None,
                "remainingPlausibleHypotheses": None,
                "proposedNextTest": "Execute chronologically locked L07 rounds and stop for human review.",
                "informationGainRationale": "These families isolate three missing-config mechanisms and can distinguish a denominator change from a dynamical parameter change.",
                "appendOnly": True,
            },
            {
                "ledgerSequence": sequence + 2,
                "timestampUtc": timestamps[1],
                "loopId": "S19-L07",
                "recordPhase": "POST_LOOP_OCCUPANCY_MATCH_HUMAN_REVIEW_BOUNDARY",
                "beliefBeforeLoop": "No single source-grounded explanation had yet brought both branches near 88% under an explicit occupancy-only test.",
                "motivatingEvidence": "The locked adaptive rounds were motivated only by earlier L07 occupancy results and were registered before their outcomes.",
                "failureOrAmbiguityTargeted": "Whether 88% can be recovered through a coherent event denominator or a missing Poisson exposure rather than threshold-only tuning.",
                "selectedHypotheses": "Strict H>0.9 at parent-selected-daughter boundaries and strict molecular H>0.9 at h=2.875; threshold variants retained as exploratory comparators.",
                "learned": "Frozen original-exposure fission boundaries gave 0.8772/0.8777. All-molecular h=2.875 gave 0.8811/0.8810 and fresh-seed 0.8838/0.8803. The high-h route shortened trajectories and persistence markedly; boundary projection retained paper-scale persistence more closely.",
                "weakenedHypotheses": "Exposure values through 1.25 cannot explain 88% on the all-molecular clock; occupancy alone cannot identify author settings; threshold-only matching is outcome-directed and nonunique.",
                "remainingPlausibleHypotheses": "A generation/fission inheritance denominator is the most paper-coherent lead; an omitted h around 2.875 is a reproducible occupancy-only alternative; exact author semantics remain unresolved.",
                "proposedNextTest": "Mandatory human review. If continued, freeze a discriminating untouched test of boundary denominator versus exposure using trajectory-length/persistence and source fingerprints; do not auto-start it.",
                "informationGainRationale": "A follow-up must distinguish two mechanisms that both match occupancy, rather than create additional opportunities to match 88%.",
                "appendOnly": True,
            },
        ]
    )
    write_parquet(ledger_path, _append_unique(ledger, ledger_additions, "ledgerSequence"))

    md_path = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    text = md_path.read_text(encoding="utf-8")
    marker = "## Entry 015 — S19-L07 occupancy-only search lock"
    if marker in text:
        raise RuntimeError("S19-L07 Markdown self-improvement entries already exist")
    text += """

## Entry 015 — S19-L07 occupancy-only search lock

- **Belief before:** the 98%-versus-88% discrepancy could reflect the molecular clock or denominator, a threshold/transcription discrepancy, or an omitted Poisson exposure.
- **Human waiver:** closeness to approximately 88% was the sole scientific success target; exact 88% and all other paper-fingerprint/promotion gates were waived, but logging, provenance, regeneration, numerical validity, and artifact integrity were not.
- **Selected hypotheses:** clock/boundary interpretations, a fixed threshold/transcription family, a Poisson-exposure sweep, adaptive local bracketing, and one fresh-seed validation.
- **Why the search could add information:** it separated event-definition and simulator-setting explanations while retaining every unsuccessful setting.

## Entry 016 — S19-L07 learning and human-review boundary

- **What was learned:** strict `H>0.9` on frozen parent-to-selected-daughter fission boundaries yielded `0.8772/0.8777`; strict all-molecular `H>0.9` at `h=2.875` yielded `0.8811/0.8810`, then `0.8838/0.8803` on 100 fresh matrices.
- **What was weakened:** the tested exposure range through `h=1.25` cannot explain 88% on the all-molecular clock. Occupancy-only threshold matching is nonunique and does not identify the authors' code.
- **What remains plausible:** a fission/generation inheritance denominator is the most paper-coherent lead; omitted exposure near `h=2.875` is a reproducible occupancy-only alternative.
- **Critical distinction:** high exposure reduced trajectory length and persistence to about 284 positive steps, while original-exposure boundary projection produced about 734/779 positive molecular steps, closer to the paper's 716 control persistence. Other fingerprints remain descriptive under the waiver.
- **Next action:** mandatory human review. No L08, S20, E02, author contact, or report generation is active.
- **Why a later test could add information:** it must discriminate the boundary-denominator and exposure mechanisms on untouched evidence, not merely optimize occupancy again.
"""
    md_path.write_text(text, encoding="utf-8")

    source_report = ARTIFACT_ROOT / "source_search_report.md"
    source_text = source_report.read_text(encoding="utf-8")
    source_marker = "## S19-L07 additive source refresh — 88% versus 98% occupancy"
    if source_marker in source_text:
        raise RuntimeError("S19-L07 source report entry already exists")
    source_text += """

## S19-L07 additive source refresh — 88% versus 98% occupancy

The target paper reports `88±3%` for control self-replication probability, describes self-replicators as recurring composition-space attractors inherited across generations, and says growth uses stochastic Poisson updates. It does not state the Poisson exposure duration or uniquely define the event denominator used for the reported probability. Historical GARD v10 directly records one composition per generation and sets a drift/non-drift cutoff of `H=0.9`; a separate historical clustering helper uses `0.95`. Neither lineage identifies the target-paper implementation, and neither grounds an outcome-selected `0.97` threshold. L07 therefore treats fission-boundary occupancy, threshold transcription, and exposure duration as exploratory missing-configuration hypotheses. No author was contacted, and unlicensed source was not redistributed.
"""
    source_report.write_text(source_text, encoding="utf-8")


def _write_loop_products() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = pd.read_parquet(LOOP_ROOT / "setting_registry.parquet")
    fingerprints = _all_round_tables("trajectory_results.parquet")
    results = _all_round_tables("occupancy_results.parquet")
    rankings = _all_round_tables("occupancy_ranking.csv", reader="csv")
    rankings["rankWithinRound"] = rankings.groupby("roundId")[
        "maximumAbsoluteTargetError"
    ].rank(method="first").astype(int)
    rankings = rankings.sort_values(
        ["maximumAbsoluteTargetError", "meanAbsoluteTargetError", "settingPairId"],
        kind="stable",
    ).reset_index(drop=True)
    rankings.insert(0, "globalOccupancyRank", np.arange(1, len(rankings) + 1))

    write_parquet(LOOP_ROOT / "trajectory_fingerprint_results.parquet", fingerprints)
    write_parquet(LOOP_ROOT / "occupancy_results.parquet", results)
    write_csv(LOOP_ROOT / "occupancy_setting_ranking.csv", rankings)
    write_parquet(LOOP_ROOT / "results.parquet", results)
    write_csv(LOOP_ROOT / "candidate_ranking.csv", rankings)
    write_parquet(LOOP_ROOT / "specification_ledger.parquet", registry)
    write_parquet(LOOP_ROOT / "execution_status.parquet", registry)

    clock = fingerprints.loc[fingerprints["roundId"].eq("R01_BOUNDARY_CLOCK")].copy()
    threshold = fingerprints.loc[
        fingerprints["roundId"].eq("R02_THRESHOLD_TRANSCRIPTION")
    ].copy()
    simulator = fingerprints.loc[
        fingerprints["roundId"].isin(ROUND_IDS[2:])
    ].copy()
    write_parquet(LOOP_ROOT / "clock_boundary_results.parquet", clock)
    write_parquet(LOOP_ROOT / "threshold_sensitivity_results.parquet", threshold)
    write_parquet(LOOP_ROOT / "simulator_setting_results.parquet", simulator)

    seed_frames = []
    summary_frames = []
    for round_id in ROUND_IDS[2:]:
        seed_frames.append(pd.read_parquet(LOOP_ROOT / f"{round_id}_seed_manifest.parquet"))
        summary_frames.append(
            pd.read_parquet(LOOP_ROOT / f"{round_id}_simulation_summary.parquet")
        )
    seeds = pd.concat(seed_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    write_parquet(LOOP_ROOT / "seed_manifest.parquet", seeds)
    write_parquet(LOOP_ROOT / "simulation_runtime_and_identity_results.parquet", summaries)

    selected_pairs = [
        ("R01_BOUNDARY_CLOCK", "PARENT_DAUGHTER_BOUNDARY_ONLY_H900", "FROZEN_BOUNDARY_EVENT"),
        ("R01_BOUNDARY_CLOCK", "PARENT_DAUGHTER_OUTGOING_ELIGIBLE_H900", "FROZEN_BOUNDARY_PROJECTED"),
        ("R02_THRESHOLD_TRANSCRIPTION", "C0_AVG_H9700", "OUTCOME_GUIDED_THRESHOLD_COMPARATOR"),
        ("R05_EXPOSURE_LOCAL_BRACKETING", "L07-R05-H28750::C1_INCOMING_ALL_H900", "ADAPTIVE_EXPOSURE_BRACKET"),
        ("R06_FRESH_SEED_VALIDATION", "L07-R06-H28750::C1_INCOMING_ALL_H900", "FRESH_EXPOSURE_VALIDATION"),
        ("R06_FRESH_SEED_VALIDATION", "L07-R06-H32500::PARENT_DAUGHTER_BOUNDARY_H900", "FRESH_BOUNDARY_VALIDATION"),
    ]
    strong = []
    for round_id, pair_id, role in selected_pairs:
        subset = _selected_rows(results, round_id, pair_id).copy()
        subset.insert(0, "evidenceRole", role)
        strong.append(subset)
    strongest = pd.concat(strong, ignore_index=True)
    write_parquet(LOOP_ROOT / "strongest_setting_validation.parquet", strongest)

    controls = pd.DataFrame(
        [
            {
                "controlId": "IMMUTABLE_PRIOR_BASELINE",
                "controlType": "ARTIFACT_HASH_REPLAY",
                "passed": True,
                "details": "1584/1584 immutable prior files matched at preparation",
            },
            {
                "controlId": "FROZEN_H_AND_LABEL_REPLAY",
                "controlType": "INPUT_REPLAY",
                "passed": True,
                "details": "200 trajectories and 180635 selected-clock rows; exact labels; scores within amendment 001",
            },
            {
                "controlId": "R03_EXACT_S13Y_TRAJECTORY_REPLAY",
                "controlType": "SIMULATOR_REPLAY",
                "passed": bool(pd.read_parquet(LOOP_ROOT / "R03_exact_s13y_replay.parquet")["passed"].all()),
                "details": "200/200 exact trajectory hashes",
            },
            {
                "controlId": "R06_FRESH_SEED_FIREWALL",
                "controlType": "SEED_AND_MATRIX_INDEPENDENCE",
                "passed": json.loads((LOOP_ROOT / "R06_seed_firewall.json").read_text())["passed"],
                "details": "100 fresh beta and 100 fresh initial-state hashes; zero overlap with S13Y",
            },
            {
                "controlId": "FULL_EXACT_REGENERATION",
                "controlType": "DETERMINISTIC_REGENERATION",
                "passed": json.loads((LOOP_ROOT / "regeneration_validation.json").read_text())["passed"],
                "details": "20/20 scientific table components across six rounds replayed exactly",
            },
        ]
    )
    write_parquet(LOOP_ROOT / "negative_control_results.parquet", controls)

    robustness = strongest[
        [
            "evidenceRole",
            "roundId",
            "settingPairId",
            "candidateId",
            "matrixCount",
            "meanOccupancy",
            "ci025MeanOccupancy",
            "ci975MeanOccupancy",
            "absoluteTargetError",
            "withinPaperApproximateBand",
        ]
    ].copy()
    write_parquet(LOOP_ROOT / "robustness_results.parquet", robustness)

    failures = pd.DataFrame(
        [
            {
                "failureId": "L07-F001",
                "phase": "PREANALYSIS_REPLAY_ATTEMPT_001",
                "status": "RECOVERED_BY_PROSPECTIVE_VALUE_PRESERVING_AMENDMENT",
                "failureType": "BIT_EXACT_FLOAT64_SCORE_REPLAY",
                "message": "All 180635 boolean labels matched, but equivalent normalization orders differed by <=8.881784197001252e-16 and <=8 ULP; no scientific L07 occupancy was opened.",
                "scientificValueEligible": False,
                "repair": "Amendment 001 required exact masks/labels plus abs<=1e-12, rel<=1e-12, ULP<=8; all subsequent replay passed.",
            }
        ]
    )
    write_csv(LOOP_ROOT / "failure_ledger.csv", failures)

    bundles = {
        "schema": "eidosoma.e01.s19_l07_candidate_bundle_registry.v1",
        "loopId": "S19-L07",
        "soleScientificTarget": "CLOSENESS_TO_APPROXIMATELY_0.88_OCCUPANCY",
        "rounds": [],
    }
    for round_id in ROUND_IDS:
        status = json.loads((LOOP_ROOT / f"{round_id}_status.json").read_text())
        bundles["rounds"].append(
            {
                "roundId": round_id,
                "settingCount": int(registry["roundId"].eq(round_id).sum()),
                "status": "COMPLETE",
                "outcomeAccessed": True,
                "resultPath": str(LOOP_ROOT / f"{round_id}_occupancy_results.parquet"),
                "trajectoryResultCount": status["trajectoryResultCount"],
            }
        )
    (LOOP_ROOT / "candidate_bundle_registry.yaml").write_text(
        yaml.safe_dump(bundles, sort_keys=False), encoding="utf-8"
    )

    _make_figure(results, registry, fingerprints)
    return results, fingerprints, summaries


def _write_reports(
    results: pd.DataFrame, fingerprints: pd.DataFrame, summaries: pd.DataFrame
) -> None:
    def occ(round_id: str, pair: str) -> list[float]:
        return _selected_rows(results, round_id, pair)["meanOccupancy"].astype(float).tolist()

    def fp(round_id: str, pair: str, columns: list[str]) -> pd.DataFrame:
        subset = fingerprints.loc[
            fingerprints["roundId"].eq(round_id)
            & fingerprints["settingPairId"].eq(pair)
        ]
        return subset.groupby("candidateId")[columns].mean(numeric_only=True)

    boundary_occ = occ("R01_BOUNDARY_CLOCK", "PARENT_DAUGHTER_BOUNDARY_ONLY_H900")
    projected_occ = occ("R01_BOUNDARY_CLOCK", "PARENT_DAUGHTER_OUTGOING_ELIGIBLE_H900")
    bracket_occ = occ(
        "R05_EXPOSURE_LOCAL_BRACKETING", "L07-R05-H28750::C1_INCOMING_ALL_H900"
    )
    fresh_occ = occ(
        "R06_FRESH_SEED_VALIDATION", "L07-R06-H28750::C1_INCOMING_ALL_H900"
    )
    boundary_fp = fp(
        "R01_BOUNDARY_CLOCK",
        "PARENT_DAUGHTER_BOUNDARY_ONLY_H900",
        ["persistence", "consistency", "firstOnsetRawStep1", "episodeCount"],
    )
    projected_fp = fp(
        "R01_BOUNDARY_CLOCK",
        "PARENT_DAUGHTER_OUTGOING_ELIGIBLE_H900",
        ["persistence", "consistency", "firstOnsetRawStep1", "episodeCount"],
    )
    threshold_fp = fp(
        "R02_THRESHOLD_TRANSCRIPTION",
        "C0_AVG_H9700",
        ["occupancy", "persistence", "consistency", "firstOnsetRawStep1"],
    )
    fresh_fp = fp(
        "R06_FRESH_SEED_VALIDATION",
        "L07-R06-H28750::C1_INCOMING_ALL_H900",
        ["analysisUnitCount", "persistence", "consistency", "firstOnsetRawStep1"],
    )
    replay = json.loads((LOOP_ROOT / "preanalysis_replay_validation.json").read_text())
    regen = json.loads((LOOP_ROOT / "regeneration_validation.json").read_text())
    initial_fail = json.loads(
        (LOOP_ROOT / "preanalysis_replay_validation_attempt_001_failed.json").read_text()
    )
    registry = pd.read_parquet(LOOP_ROOT / "setting_registry.parquet")
    scientific_worker_cpu = float(summaries["cpuSeconds"].sum())
    round_statuses = [
        json.loads((LOOP_ROOT / f"{round_id}_status.json").read_text())
        for round_id in ROUND_IDS
    ]
    coordinator_cpu = float(sum(item["coordinatorCpuSeconds"] for item in round_statuses))

    report = f"""# E01/S19-L07 — Exploratory 88%-versus-98% occupancy-setting search

## Concise top summary

- **Research step ID:** S19-L07 (`{VERSIONED_ID}`)
- **Completion status:** COMPLETE; mandatory post-L07 human-review boundary active
- **Artifacts written:** six chronologically locked round packages; 228-setting specification/attempt ledger; 29,200 trajectory fingerprints; occupancy, clock/boundary, threshold, exposure, fresh-seed, seed, runtime, validation, figure, failure, classification, status, provenance, and hash manifests; this canonical report and one-page handoff
- **Validation result:** PASS — 1,584/1,584 immutable prior files; 200 trajectories and 180,635 frozen clock/label rows; 200/200 exact S13Y simulator replays; zero fresh-seed hash overlap; all 228 settings accounted for; 20/20 independently regenerated scientific table components exact; repository pushed and clean
- **Outcome classification:** `EXPLORATORY_PAPER_MATCH` — occupancy only; the sole human-waived target passed in both branches through two nonunique mechanisms
- **Caveats or blockers:** occupancy alone cannot identify author code; the search is adaptive; exact paper denominator, exposure, clock, and clustering semantics remain unavailable; high exposure shortens trajectories markedly; threshold matching is outcome-directed; other temporal/predictive/causal gates were waived, not passed
- **Lay summary:** We can now reproduce approximately 88% in both branches. The most paper-coherent route is to count inheritance at fission boundaries (`87.72%/87.77%`) rather than every smooth molecular update. A second route uses the unchanged molecular `H>0.9` label but a previously undocumented Poisson exposure near `h=2.875` (`88.38%/88.03%` on fresh matrices). These are reproducible explanations, not proof of the authors' exact settings.
- **Recommended next action:** Human review. If another loop is authorized, freeze one untouched discriminating test between the fission-boundary denominator and high-exposure mechanisms using trajectory-length/persistence and source fingerprints. Do not start L08, S20, E02, author contact, or report-bundle generation automatically.

## Frozen question and waiver

L07 asked only which paper- or source-plausible settings can move self-replicator occupancy from the frozen adjacent-molecular value near 98% toward the paper's approximately 88%. The human explicitly made occupancy proximity the sole scientific success target, did not require exact 88%, and waived every other paper-fingerprint and promotion gate. The waiver did **not** relax complete attempt logging, candidate separation, preservation of prior evidence, numerical validity, provenance, deterministic regeneration, or artifact integrity.

No L07 result is confirmatory. No result is labelled author-code identity. No emergence, prediction, intervention, or causal-control result was used to select settings.

## Lay interpretation

The 98% value is not an unavoidable property of these simulations. It is produced by asking, at every molecular observation, whether composition is very similar to its immediate predecessor. Three changes can lower it toward 88%:

1. Count the parent-to-selected-daughter inheritance event once per fission. This retains strict `H>0.9`, the frozen trajectories, and the original exposures, and gives `87.72%/87.77%`.
2. Retain every molecular observation and strict `H>0.9`, but use larger Poisson batches. A stable region near `h=2.875` gives `88.11%/88.10%` in the bracketing run and `88.38%/88.03%` on fresh matrices.
3. Raise or change the similarity transcription. Thresholds near `0.97` can also give about 88%, but this was outcome-directed and is not the historical source's `0.9` setting.

The first explanation better preserves the paper's generational inheritance language and paper-scale persistence. The second proves that an omitted simulator parameter can reproduce occupancy, but it also shortens the trajectories, so it is not by itself a coherent reconstruction of Table 1.

![Occupancy reconstruction summary](occupancy_search_summary.png)

**Figure 1.** Exploratory occupancy paths. Left: all-molecular strict incoming `H>0.9` occupancy falls as Poisson exposure grows; stars are the fresh-seed `h=2.875` validation. Right: multiple settings reach the paper's approximate occupancy band, demonstrating nonidentifiability from occupancy alone.

## Inputs and immutable context

- Original paper attachment: `pdf-markdown.md`, arXiv version `2607.28250v1`.
- Frozen S13Y data: 100 shared catalytic matrices, 200 candidate-2/candidate-3 trajectories, selected molecular clocks, adjacent-H arrays, and exact `H>0.9` labels.
- Candidate 2 baseline: `h=0.6031526490073492`, first-daughter continuation.
- Candidate 3 baseline: `h=0.5613315384859516`, random-nonempty daughter continuation.
- Historical GARD source commit: `86dff6320d5ae91b4e831471079ff46749b14df9`; source retained in cache only because no license file was found.
- Immutable baseline: all S01–S18, V1/V2, S19-L01–L06R, classifications, failures, and the S17 waiver.

The paper directly reports `88±3%` for control probability, describes recurring composition-space attractors inherited across generations, and specifies Poisson updates. It does not uniquely state the probability denominator/object or Poisson exposure duration. Historical GARD v10 uses generation-level compositions and a drift/non-drift `H=0.9` parameter; another clustering helper uses `0.95`. Those sources are lineage evidence, not target-paper code.

## Detailed methods and chronological search

### Numerical gate and amendment

The first replay attempt opened no L07 occupancy. All 180,635 frozen boolean labels agreed, but two mathematically equivalent float64 normalization orders were not bit-identical (maximum absolute `{initial_fail.get('maximumAbsoluteError', 8.881784197001252e-16)}`). A separately committed value-preserving amendment required identical finite masks and labels plus absolute and relative error `<=1e-12` and ULP distance `<=8`. The amended replay passed all 200 trajectories: maximum absolute `{replay['maximumAbsoluteScoreError']}`, relative `{replay['maximumRelativeScoreError']}`, and `{replay['maximumUlpDistance']}` ULP.

### Round inventory

| Round | Frozen question | Registered settings | Outcome |
| --- | --- | ---: | --- |
| R01 | Clock, boundary object, strict/`>=`, and fixed projections | 16 | Parent→selected-daughter strict `H>0.9` boundary occupancy `0.8772/0.8777` |
| R02 | Fixed threshold/transcription family | 48 | Several `~0.97` settings enter the band; nonunique and outcome-directed |
| R03 | Coarse Poisson exposure and exact frozen replay | 100 | All-molecular occupancy declines from about 0.986 at `h=0.45` to about 0.956 at `h=1.25`; boundary occupancy remains near 0.88 |
| R04 | Missing-exposure diagnostic through `h=4` | 36 | All-molecular match appears near `h=3`; boundary match remains broad |
| R05 | Local bracket `h=2.75–3.25` | 20 | `h=2.875` gives `0.881090/0.880995` all-molecular occupancy |
| R06 | Fresh 100-matrix seed set | 8 | `h=2.875` validates at `0.883845/0.880294`; zero matrix/initial overlap |

Every setting was serialized before its result in `setting_registry.parquet` and retained after outcome access in `chronological_attempt_ledger.parquet`. Unsuccessful settings were not deleted or reordered. Candidate/branch results are separate; pooled selection was not used.

### Simulation contract

The exposure rounds retained the S13Y simulation kernel, 100 molecule types, 100 fissions, fixed shared matrices/initial states per round, overshoot handling that trims only excess newly joined molecules, and the two continuation rules. CPU float64 was authoritative; no GPU was used. R06 used a new domain-separated 256-bit root and required no beta or initial-state hash overlap with S13Y.

### Occupancy and uncertainty

For each trajectory, occupancy is the fraction of eligible analysis units labelled positive. Each catalytic matrix is one inferential unit. Reported intervals are deterministic 4,096-replicate matrix-bootstrap 95% intervals. Pair ranking minimizes the maximum candidate/branch absolute error from `0.88`, then mean error. The approximate target band is `0.88±0.03`, solely for L07.

## Results

### 1. Fission-boundary denominator matches 88% without changing trajectories

| Definition on frozen original-exposure trajectories | Candidate 2 | Candidate 3 | Key scope |
| --- | ---: | ---: | --- |
| Adjacent molecular incoming `H>0.9` | 0.980891 | 0.982657 | Every selected molecular observation |
| Parent→selected-daughter boundary, strict `H>0.9` | {boundary_occ[0]:.6f} | {boundary_occ[1]:.6f} | 100 fission events/run |
| Boundary label projected to following eligible interval | {projected_occ[0]:.6f} | {projected_occ[1]:.6f} | Molecular time, first prefix excluded |

The boundary-only result is the closest paper/source-plausible match because it changes neither threshold nor simulator and directly measures inheritance through fission. It also remains near 88% across every tested exposure and continuation rule; therefore its match is not an `h`-selected accident.

The denominator matters. Boundary-only persistence is only the number of positive fissions (`{boundary_fp.iloc[0]['persistence']:.2f}/{boundary_fp.iloc[1]['persistence']:.2f}`), so it is not directly comparable to the paper's molecular-step persistence. Projecting those boundary decisions onto molecular intervals gives persistence `{projected_fp.iloc[0]['persistence']:.2f}/{projected_fp.iloc[1]['persistence']:.2f}`, close in scale to the paper control value 716, but occupancy is about 86.5%, consistency is `{projected_fp.iloc[0]['consistency']:.3f}/{projected_fp.iloc[1]['consistency']:.3f}`, and onset is `{projected_fp.iloc[0]['firstOnsetRawStep1']:.2f}/{projected_fp.iloc[1]['firstOnsetRawStep1']:.2f}` steps. Those descriptive mismatches do not fail the human-waived L07 target, but they prevent an exact label claim.

### 2. A missing Poisson exposure also matches 88%

The coarse sweep showed a smooth occupancy decrease under the all-molecular strict label. Values up to the earlier S12F ceiling `h=1.25` remained at 95.5–95.6%. Extending the paper-undocumented exposure reached the target near `h=3`; the fixed bracket selected `h=2.875` for cross-branch occupancy proximity.

| Dataset | First-daughter branch | Random-nonempty branch | Maximum error from 0.88 |
| --- | ---: | ---: | ---: |
| R05 shared S13Y matrices, `h=2.875` | {bracket_occ[0]:.6f} | {bracket_occ[1]:.6f} | {max(abs(value-0.88) for value in bracket_occ):.6f} |
| R06 fresh matrices, `h=2.875` | {fresh_occ[0]:.6f} | {fresh_occ[1]:.6f} | {max(abs(value-0.88) for value in fresh_occ):.6f} |

The fresh validation passed the seed firewall: 100 new catalytic-matrix hashes and 100 new initial-state hashes, with zero overlap. This makes `h=2.875` a reproducible occupancy-only lead.

It is not a complete Table-1 reconstruction. On fresh matrices, mean selected-clock length was `{fresh_fp.iloc[0]['analysisUnitCount']:.2f}/{fresh_fp.iloc[1]['analysisUnitCount']:.2f}` and persistence only `{fresh_fp.iloc[0]['persistence']:.2f}/{fresh_fp.iloc[1]['persistence']:.2f}`; onset remained `{fresh_fp.iloc[0]['firstOnsetRawStep1']:.2f}/{fresh_fp.iloc[1]['firstOnsetRawStep1']:.2f}` and consistency `{fresh_fp.iloc[0]['consistency']:.3f}/{fresh_fp.iloc[1]['consistency']:.3f}`. Large Poisson batches also produced substantially larger overshoot before the frozen trim rule. Thus high exposure reaches 88% partly by reducing the number and smoothness of recorded updates, and the paper gives no evidence that `h=2.875` was used.

### 3. Threshold/transcription routes are nonunique

The best cross-candidate incoming-clock threshold pair was C0 strict `H>0.9725` (`0.876139/0.886973`). A historical-technique-like two-neighbor average on C0 at `H>0.97` gave occupancy `{threshold_fp.iloc[0]['occupancy']:.6f}/{threshold_fp.iloc[1]['occupancy']:.6f}`, persistence `{threshold_fp.iloc[0]['persistence']:.2f}/{threshold_fp.iloc[1]['persistence']:.2f}`, and consistency `{threshold_fp.iloc[0]['consistency']:.3f}/{threshold_fp.iloc[1]['consistency']:.3f}`. That consistency is directionally closer to the paper's 0.38 than the adjacent incoming label, but onset remained only `{threshold_fp.iloc[0]['firstOnsetRawStep1']:.2f}/{threshold_fp.iloc[1]['firstOnsetRawStep1']:.2f}` steps.

These results confirm the user's prior observation: a threshold can force occupancy toward 88% without recovering the temporal state. Moreover, `0.97` was searched after the target was known, conflicts with the historical `0.9` parameter, and is not promoted as an author setting.

### 4. What most likely explains 88% versus 98%

The evidence now favors a **measurement-object/denominator mismatch** as the most parsimonious source-grounded explanation: the paper discusses recurrence and inheritance across generations, while the frozen 98% label evaluates smoothness at every molecular observation. Measuring strict similarity at parent→daughter fission events immediately produces 87.7% under both original candidate pipelines. This inference is stronger than a bare numerical match but remains an inference; the paper also describes recurrence relative to a recurring attractor, which a single parent→daughter comparison does not fully implement.

An omitted exposure near `h=2.875` is a genuine alternative explanation for occupancy alone. It is less coherent with molecular-step persistence and has no recovered source identity. Occupancy therefore does not identify one unique author pipeline.

## Validation

| Check | Result |
| --- | --- |
| Immutable S01–S18, V1/V2, L01–L06R baseline | PASS, 1,584/1,584 files |
| Frozen trajectory/clock/H/label replay | PASS, 200 trajectories and 180,635 rows; exact labels |
| Numerical amendment | PASS, masks/labels exact; abs/rel `<=1e-12`, ULP `<=8` |
| Exact frozen S13Y simulator replay in R03 | PASS, 200/200 trajectory hashes |
| Setting accounting | PASS, {len(registry)}/{len(registry)} registered settings complete |
| Fresh R06 seed firewall | PASS, 0 beta and 0 initial-state overlaps |
| Independent regeneration | PASS, {regen['passedComponentCount']}/{regen['componentCount']} scientific components exact |
| Repository | PASS, pushed `699bdfa08696a2a5d5e4f83e441f60884303e2b9`, clean at regeneration |
| Scientific failures | None after replay amendment; initial failed attempt retained as `L07-F001` |

The NumPy warnings recorded during fingerprints occur when consecutive-label Pearson correlation is undefined for constant sequences; the code serializes those values as status-bearing nulls. They do not affect occupancy or any success decision.

## Runtime, commands, and dependencies

Core commands:

```text
PYTHONPATH=src:. pytest -q tests/e01/test_s19_l07.py
PYTHONPATH=src python scripts/e01/run_s19_l07.py prepare
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r01 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r02 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r03 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py prepare-r04 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r04 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py prepare-r05 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r05 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py prepare-r06 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r06 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py regenerate --workers 8
```

The six outcome rounds used `{scientific_worker_cpu/3600:.4f}` worker CPU-hours plus `{coordinator_cpu/3600:.4f}` coordinator CPU-hours; regeneration took `{regen['wallSeconds']/60:.2f}` wall-minutes. Eight workers and one numerical-library thread per worker were used. GPU hours were zero. Dependencies were the preinstalled Python 3.13, NumPy, pandas, SciPy, PyArrow, PyYAML, and Matplotlib stack; no package was installed.

Pushed commits, in order: `a3ef41d` preregistration; `edf9851` numerical amendment; `13efb7e` exposure extension; `d58becf` local bracket; `d6f22bb` fresh validation; `699bdfa` exact regeneration audit.

## Provenance and artifact map

- Governing locks: `preregistration.yaml`, `method_lock.json`, `round_registry.yaml`, and `round_R04/R05/R06_lock.yaml`.
- Complete setting history: `setting_registry.parquet`, `chronological_attempt_ledger.parquet`, and `specification_ledger.parquet`.
- Primary machine result: `occupancy_results.parquet`; full per-trajectory evidence: `trajectory_fingerprint_results.parquet`.
- Mechanism-specific evidence: `clock_boundary_results.parquet`, `threshold_sensitivity_results.parquet`, and `simulator_setting_results.parquet`.
- Fresh validation: `R06_FRESH_SEED_VALIDATION_*`, `R06_seed_firewall.json`, and `strongest_setting_validation.parquet`.
- Exact regeneration: `regeneration_results.parquet` and `regeneration_validation.json`.
- Integrity: `immutable_prior_postcheck.json`, `storage_validation.json`, and `artifact_manifest.json`.
- Reproducible repository code: `src/e01_s19_occupancy_search/`, `scripts/e01/run_s19_l07.py`, this finalizer, configs, and tests on branch `eidosoma/groups/42`.

## Caveats and limitations

1. The search was explicitly adaptive and used a known 88% target. An occupancy match is exploratory.
2. The paper's 88% appears in Table 1's control intervention context; equivalence to the frozen S13Y baseline dataset is plausible but not established by author code.
3. Boundary-only occupancy changes the unit from molecular steps to fissions. Projection restores molecular time but requires an undocumented rule.
4. High exposure is paper-plausible only because exposure duration is omitted; no source identifies `h=2.875`, and high batches alter trajectory length and overshoot.
5. Threshold values near 0.97 are outcome-selected and not a substitute for the historical/source `0.9` contract.
6. None of these routes reconstructs the full recurring-attractor definition uniquely. Full-run compotype and recurrence alternatives remain historical exploratory evidence from L02–L06R.
7. Other temporal fingerprints were descriptive by human waiver. Prediction and causal-control non-support from S16–S18 is unchanged.
8. No authors were contacted, and public code without an identified license was not redistributed.

## Outcome and mandatory handoff

L07 succeeds on its sole authorized target: there are reproducible, candidate-consistent settings close to 88%. The result is an `EXPLORATORY_PAPER_MATCH` limited to occupancy. The strongest paper-coherent lead is the strict parent→selected-daughter fission-boundary denominator; the strongest all-molecular alternative is an omitted fixed exposure near `h=2.875`. Neither is confirmed as the authors' implementation.

All L07 artifacts are frozen for human review. No L08, S20, E02, author contact, or report-bundle generation has been activated.
"""
    (LOOP_ROOT / "S19_L07_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")

    decision = f"""# S19-L07 one-page decision summary

## Concise top summary

- **Research step ID:** S19-L07
- **Completion status:** COMPLETE; stopped at mandatory human review
- **Artifacts written:** six locked round packages, complete 228-setting ledger, machine results, validation/hash manifests, figure, canonical full report, and this summary
- **Validation result:** PASS — immutable baseline, frozen replay, fresh-seed firewall, all-setting accounting, and 20/20 exact regeneration components
- **Outcome classification:** `EXPLORATORY_PAPER_MATCH` limited to occupancy
- **Caveats or blockers:** adaptive known-target search; multiple nonunique settings; author implementation unavailable; other temporal/predictive/causal gates waived
- **Recommended next action:** Human review; do not automatically activate L08 or S20

## Decision-relevant result

L07 found two reproducible explanations for the 88%-versus-98% gap:

1. **Fission-boundary denominator:** on the untouched frozen candidate trajectories, strict parent→selected-daughter `H>0.9` is `87.72%/87.77%`. It changes no trajectory, threshold, or exposure and fits the paper's generational-inheritance language. Projecting it to molecular time gives persistence about `734/779`, near the paper control's `716`, though consistency and onset remain different.
2. **Missing exposure:** all-molecular strict `H>0.9` at `h=2.875` is `88.11%/88.10%`; on 100 fresh matrices it is `88.38%/88.03%`. It is numerically reproducible but shortens trajectories, giving persistence only about `284`, so occupancy alone is a weaker coherent reconstruction.

Threshold/transcription variants near `0.97` also match occupancy, but they were selected against a known target and conflict with the source `0.9` setting.

## Interpretation boundary

The result identifies plausible settings, not author-code identity. It does not revise S18, establish a recurring-attractor label, restore S16 prediction, or establish S17 causal control.

## Human-review choices

- Continue S19 with a separately authorized untouched discrimination of the boundary-denominator and exposure leads.
- Activate an S20 mode under its existing rules.
- Pause E01.

No option is active automatically.
"""
    (LOOP_ROOT / "loop_decision_summary.md").write_text(decision, encoding="utf-8")


def _write_status_and_manifests(
    results: pd.DataFrame, fingerprints: pd.DataFrame, summaries: pd.DataFrame
) -> None:
    runtime_statuses = [
        json.loads((LOOP_ROOT / f"{round_id}_status.json").read_text())
        for round_id in ROUND_IDS
    ]
    regen = json.loads((LOOP_ROOT / "regeneration_validation.json").read_text())
    runtime = {
        "schema": "eidosoma.e01.s19_l07_runtime_manifest.v1",
        "workers": 8,
        "threadsPerWorker": 1,
        "gpuHours": 0,
        "rounds": runtime_statuses,
        "scientificWorkerCpuSeconds": float(summaries["cpuSeconds"].sum()),
        "scientificTrajectoryWallSecondsSummed": float(summaries["wallSeconds"].sum()),
        "coordinatorCpuSeconds": float(sum(item["coordinatorCpuSeconds"] for item in runtime_statuses)),
        "regenerationWallSeconds": regen["wallSeconds"],
        "maximumCpuHourCeiling": 100,
        "maximumGpuHourCeiling": 12,
        "withinCeiling": True,
        "numericalAuthority": "CPU_FLOAT64",
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)

    classification = {
        "schema": "eidosoma.e01.s19_l07_classification.v1",
        "researchStepId": "S19-L07",
        "versionedId": VERSIONED_ID,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "topLevelClassification": "EXPLORATORY_PAPER_MATCH",
        "classificationScope": "OCCUPANCY_ONLY",
        "soleScientificTarget": "CLOSENESS_TO_APPROXIMATELY_0.88_OCCUPANCY",
        "soleScientificTargetPassed": True,
        "exact88Required": False,
        "otherPromotionAndFingerprintGatesWaivedNotPassed": True,
        "leadClassifications": [
            {
                "leadId": "FISSION_BOUNDARY_DENOMINATOR",
                "classification": "EXPLORATORY_PAPER_MATCH",
                "authorIdentity": False,
            },
            {
                "leadId": "ALL_MOLECULAR_H900_EXPOSURE_H2875",
                "classification": "EXPLORATORY_PAPER_MATCH",
                "authorIdentity": False,
            },
            {
                "leadId": "THRESHOLD_NEAR_H097",
                "classification": "METHOD_DEPENDENT_LEAD",
                "authorIdentity": False,
            },
        ],
        "predictionClassificationChanged": False,
        "causalControlClassificationChanged": False,
        "promotedToS20": False,
        "nextStepActive": False,
    }
    write_json(LOOP_ROOT / "classification.json", classification)
    status = {
        "researchStepId": "S19-L07",
        "stepNumber": 19,
        "success": True,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [
            str(LOOP_ROOT / "S19_L07_FULL_RESULTS.md"),
            str(LOOP_ROOT / "results.parquet"),
            str(LOOP_ROOT / "trajectory_fingerprint_results.parquet"),
            str(LOOP_ROOT / "setting_registry.parquet"),
            str(LOOP_ROOT / "regeneration_validation.json"),
            str(LOOP_ROOT / "artifact_manifest.json"),
        ],
        "validationResult": "PASS_IMMUTABLE_REPLAY_FRESH_SEED_FIREWALL_228_SETTING_ACCOUNTING_AND_20_OF_20_EXACT_REGENERATION_COMPONENTS",
        "outcomeClassification": "EXPLORATORY_PAPER_MATCH_OCCUPANCY_ONLY",
        "caveatsOrBlockers": [
            "adaptive_known_target_search",
            "occupancy_match_nonunique",
            "author_implementation_unavailable",
            "boundary_denominator_and_projection_ambiguous",
            "high_exposure_shortens_trajectory",
            "threshold_matching_outcome_guided",
            "other_temporal_prediction_and_causal_gates_waived_not_passed",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_L08_S20_E02_AUTHOR_CONTACT_OR_REPORT_BUNDLE",
    }
    write_json(LOOP_ROOT / "status.json", status)
    write_json(ARTIFACT_ROOT / "s19_status.json", status)

    postcheck = validate_prior_baseline()
    postcheck["phase"] = "POST_L07_FINALIZATION"
    write_json(LOOP_ROOT / "immutable_prior_postcheck.json", postcheck)
    if not postcheck["passed"]:
        raise RuntimeError("immutable prior postcheck failed")

    cache_root = Path("/cache/e01_s19_l07")
    retained_files = [item for item in LOOP_ROOT.rglob("*") if item.is_file()]
    retained_bytes = sum(item.stat().st_size for item in retained_files)
    temp_bytes = (
        sum(item.stat().st_size for item in cache_root.rglob("*") if item.is_file())
        if cache_root.exists()
        else 0
    )
    storage = {
        "schema": "eidosoma.e01.s19_l07_storage_validation.v1",
        "retainedFileCountBeforeManifest": len(retained_files),
        "retainedBytesBeforeManifest": retained_bytes,
        "retainedGiB": retained_bytes / 2**30,
        "retainedCeilingGiB": 25,
        "temporaryBytes": temp_bytes,
        "temporaryGiB": temp_bytes / 2**30,
        "temporaryCeilingGiB": 75,
        "passed": retained_bytes <= 25 * 2**30 and temp_bytes <= 75 * 2**30,
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("L07 storage validation failed")


def _update_root_control_files() -> None:
    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(loop_path.read_text(encoding="utf-8"))
    if any(item["loopId"] == "S19-L07" for item in registry["loops"]):
        raise RuntimeError("S19-L07 already exists in loop registry")
    registry["loops"].append(
        {
            "loopId": "S19-L07",
            "versionedLoopId": VERSIONED_ID,
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "authorized": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": True,
            "completed": True,
            "eligibleScientificResults": True,
            "soleScientificTarget": "CLOSENESS_TO_APPROXIMATELY_0.88_OCCUPANCY",
            "soleScientificTargetPassed": True,
            "classification": "EXPLORATORY_PAPER_MATCH_OCCUPANCY_ONLY",
            "nextStepActive": False,
        }
    )
    loop_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["history"].extend(
        [
            {
                "date": COMPLETED_AT,
                "decision": "AUTHORIZE_ADDITIVE_S19_L07_OCCUPANCY_ONLY_SEARCH",
                "scope": VERSIONED_ID,
                "source": "explicit_human_direction",
            },
            {
                "date": COMPLETED_AT,
                "decision": "EXPAND_L07_WITH_ADDITIONAL_LOCKED_ROUNDS_ON_OCCUPANCY_LEAD",
                "scope": "R05_LOCAL_BRACKETING_AND_R06_FRESH_SEED_VALIDATION",
                "source": "explicit_human_direction_during_active_L07",
            },
            {
                "date": COMPLETED_AT,
                "decision": "S19_L07_COMPLETE_MANDATORY_HUMAN_REVIEW",
                "scope": VERSIONED_ID,
                "source": "validated_locked_execution_result",
            },
        ]
    )
    history["pendingDecision"] = "POST_S19_L07_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def _manifest(root: Path, required: list[str], schema: str) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json" and path.parent == root:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    paths = {item["path"] for item in files}
    missing = sorted(set(required).difference(paths))
    return {
        "schema": schema,
        "root": str(root),
        "requiredFiles": required,
        "missing": missing,
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
        "passed": not missing,
    }


def _write_manifests() -> None:
    required = [
        "preregistration.yaml",
        "method_lock.json",
        "candidate_ranking.csv",
        "candidate_bundle_registry.yaml",
        "seed_manifest.parquet",
        "input_manifest.json",
        "source_snapshot_manifest.json",
        "execution_status.parquet",
        "specification_ledger.parquet",
        "results.parquet",
        "negative_control_results.parquet",
        "robustness_results.parquet",
        "failure_ledger.csv",
        "runtime_manifest.json",
        "storage_validation.json",
        "regeneration_validation.json",
        "classification.json",
        "status.json",
        "loop_decision_summary.md",
        "S19_L07_FULL_RESULTS.md",
        "research_step_full_results.md",
        "clock_boundary_results.parquet",
        "threshold_sensitivity_results.parquet",
        "simulator_setting_results.parquet",
        "occupancy_results.parquet",
        "trajectory_fingerprint_results.parquet",
        "occupancy_setting_ranking.csv",
        "strongest_setting_validation.parquet",
        "occupancy_search_summary.png",
        "immutable_prior_postcheck.json",
    ]
    loop_manifest = _manifest(
        LOOP_ROOT, required, "eidosoma.e01.s19_l07_artifact_manifest.v1"
    )
    write_json(LOOP_ROOT / "artifact_manifest.json", loop_manifest)
    if not loop_manifest["passed"]:
        raise RuntimeError(f"L07 artifact manifest missing {loop_manifest['missing']}")
    root_required = [
        "candidate_registry.parquet",
        "source_search_ledger.parquet",
        "source_search_report.md",
        "self_improvement_ledger.parquet",
        "SELF_IMPROVEMENT_LEDGER.md",
        "loop_registry.yaml",
        "human_review_history.json",
        "s19_status.json",
        "research_step_full_results.md",
        "loops/L07/artifact_manifest.json",
        "loops/L07/S19_L07_FULL_RESULTS.md",
    ]
    root_manifest = _manifest(
        ARTIFACT_ROOT, root_required, "eidosoma.e01.s19_root_artifact_manifest.v1"
    )
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", root_manifest)
    if not root_manifest["passed"]:
        raise RuntimeError(f"S19 root manifest missing {root_manifest['missing']}")


def main() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "rev-parse", "origin/eidosoma/groups/42"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status or head != remote:
        raise RuntimeError("finalization requires clean pushed repository state")
    if not json.loads((LOOP_ROOT / "regeneration_validation.json").read_text())["passed"]:
        raise RuntimeError("regeneration must pass before finalization")
    results, fingerprints, summaries = _write_loop_products()
    _append_root_ledgers()
    _write_reports(results, fingerprints, summaries)
    _write_status_and_manifests(results, fingerprints, summaries)
    _update_root_control_files()
    _write_manifests()


if __name__ == "__main__":
    main()
