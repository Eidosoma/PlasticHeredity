#!/usr/bin/env python3
"""Finalize compact evidence, figures, manifests, and report for S13X."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e01_creative_directional_search.core import EVIDENCE_CLASS, VERSION

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_ROOT = ARTIFACTS / "research_steps/S13X"
FIGURE_ROOT = STEP_ROOT / "figures"
SOURCE_VALUES = ARTIFACTS / "research_steps/S13RRR/full_source_values.parquet"
PREFIX_VALUES = ARTIFACTS / "research_steps/S13RRR/prefix_endpoint_values.parquet"

OUTCOME_CLASSIFICATION = (
    "ADAPTIVE_RETROSPECTIVE_DIRECTIONAL_RESEMBLANCE_WITH_"
    "PROSPECTIVE_AND_INTERVENTION_NON_SUPPORT"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def prior_immutability() -> dict[str, Any]:
    baseline = json.loads(
        (STEP_ROOT / "prior_artifact_baseline.json").read_text(encoding="utf-8")
    )
    mismatches = []
    for item in baseline["files"]:
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        if actual != item["sha256"]:
            mismatches.append(
                {
                    "path": str(path),
                    "expectedSha256": item["sha256"],
                    "actualSha256": actual,
                }
            )
    result = {
        "schema": "eidosoma.e01.s13x_prior_immutability_validation.v1",
        "researchStepId": "S13X",
        "fileCount": len(baseline["files"]),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    write_json(STEP_ROOT / "prior_immutability_validation.json", result)
    return result


def normalize_search_ledger() -> dict[str, Any]:
    path = STEP_ROOT / "chronological_search_ledger.csv"
    ledger = pd.read_csv(path)
    if "attemptOccurrence" in ledger.columns:
        ledger.drop(columns=["attemptOccurrence", "duplicateAttemptId"], inplace=True)
    ledger["attemptOccurrence"] = ledger.groupby("attemptId", sort=False).cumcount() + 1
    ledger["duplicateAttemptId"] = (
        ledger.groupby("attemptId")["attemptId"].transform("size") > 1
    )
    ledger.to_csv(path, index=False, lineterminator="\n")
    result = {
        "rowCount": len(ledger),
        "attemptSequenceUnique": bool(ledger["attemptSequence"].is_unique),
        "attemptSequenceMonotonic": bool(
            ledger["attemptSequence"].is_monotonic_increasing
        ),
        "duplicateExecutionRowCount": int(ledger["duplicateAttemptId"].sum()),
        "uniqueAttemptIdCount": int(ledger["attemptId"].nunique()),
        "phaseCounts": {
            str(key): int(value)
            for key, value in ledger.groupby("phase").size().items()
        },
    }
    return result


def candidate_specifications(registry: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        {
            "candidatePipelineId": "S13X-C1-PAPER-H090-MOLECULAR",
            "implementationId": "PHIRL_REGULARIZED_SOURCE",
            "metric": "emergence",
            "transform": "LEVEL",
            "labelId": "MOL_ADJACENT_INCOMING_H900",
            "alignment": "SAME_STATE",
            "role": "MOST_SOURCE_AND_PAPER_DIRECTED_RETROSPECTIVE_LEAD",
            "choiceEvidence": (
                "source-defined emergence and stated H>0.9 threshold; applying the "
                "adjacent label at molecular rather than post-fission resolution is inferred"
            ),
        },
        {
            "candidatePipelineId": "S13X-C2-TABLE-OCCUPANCY-H095",
            "implementationId": "PHIRL_REGULARIZED_SOURCE",
            "metric": "emergence",
            "transform": "LEVEL",
            "labelId": "MOL_ADJACENT_INCOMING_H950",
            "alignment": "SAME_STATE",
            "role": "ADAPTIVE_INTERMEDIATE_THRESHOLD_SENSITIVITY",
            "choiceEvidence": "outcome-guided threshold sensitivity; not an author setting",
        },
        {
            "candidatePipelineId": "S13X-C3-TABLE-OCCUPANCY-H097",
            "implementationId": "PHIRL_REGULARIZED_SOURCE",
            "metric": "emergence",
            "transform": "LEVEL",
            "labelId": "MOL_ADJACENT_INCOMING_H970",
            "alignment": "SAME_STATE",
            "role": "TABLE_OCCUPANCY_MATCHED_EXPLORATORY_LEAD",
            "choiceEvidence": (
                "outcome-guided threshold sensitivity chosen because occupancy approaches "
                "the reported 0.88; not an author setting"
            ),
        },
        {
            "candidatePipelineId": "S13X-C4-FIGURE-CAPTION-DOWNWARD-CHANGE",
            "implementationId": "PHIRL_REGULARIZED_SOURCE",
            "metric": "downwardCausation",
            "transform": "BACKWARD_DIFFERENCE",
            "labelId": "MOL_ADJACENT_AVERAGE_H970",
            "alignment": "SAME_STATE",
            "role": "BEST_CONTINUOUS_RESEMBLANCE_COMPONENT_ATOM",
            "choiceEvidence": (
                "source atom plus Figure-3-caption change interpretation and adaptive "
                "Table-1-directed label threshold"
            ),
        },
        {
            "candidatePipelineId": "S13X-C5-IIGR-EMERGENCE-COMPARATOR",
            "implementationId": "IIGR_CORRECTED_SOURCE",
            "metric": "emergence",
            "transform": "BACKWARD_DIFFERENCE",
            "labelId": "MOL_ADJACENT_AVERAGE_H970",
            "alignment": "NEXT_STATE",
            "role": "EARLIER_SOURCE_IMPLEMENTATION_COMPARATOR",
            "choiceEvidence": "best IIGR emergence member of the adaptive neighborhood",
        },
    ]
    rows = []
    for definition in definitions:
        match = registry[
            (registry["implementationId"] == definition["implementationId"])
            & (registry["metric"] == definition["metric"])
            & (registry["transform"] == definition["transform"])
            & (registry["labelId"] == definition["labelId"])
            & (registry["alignment"] == definition["alignment"])
        ]
        if len(match) != 1:
            raise RuntimeError(f"candidate pipeline lookup failed: {definition}")
        rows.append({**definition, "pipelineId": str(match.iloc[0]["pipelineId"])})
    return pd.DataFrame(rows)


def candidate_results(specifications: pd.DataFrame) -> pd.DataFrame:
    development = pd.read_csv(STEP_ROOT / "development_pipeline_results.csv")
    diagnostic = pd.concat(
        [
            pd.read_csv(STEP_ROOT / "diagnostic_pipeline_results.csv"),
            pd.read_csv(STEP_ROOT / "focused_diagnostic_results.csv"),
            pd.read_csv(STEP_ROOT / "paper_directed_label_results.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates(["pipelineId", "candidateId"], keep="last")
    inference = pd.concat(
        [
            pd.read_csv(STEP_ROOT / "diagnostic_inference.csv"),
            pd.read_csv(STEP_ROOT / "focused_diagnostic_inference.csv"),
            pd.read_csv(STEP_ROOT / "paper_directed_label_inference.csv"),
            pd.read_csv(STEP_ROOT / "anchor_diagnostic_inference.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates(["pipelineId", "candidateId"], keep="last")
    rows = []
    for spec in specifications.itertuples(index=False):
        for candidate_id in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"):
            dev = development[
                (development["pipelineId"] == spec.pipelineId)
                & (development["candidateId"] == candidate_id)
            ].iloc[0]
            diag = diagnostic[
                (diagnostic["pipelineId"] == spec.pipelineId)
                & (diagnostic["candidateId"] == candidate_id)
            ].iloc[0]
            infer = inference[
                (inference["pipelineId"] == spec.pipelineId)
                & (inference["candidateId"] == candidate_id)
            ]
            combined_defined = int(
                dev.definedTrajectoryCount + diag.definedTrajectoryCount
            )
            combined_positive = int(
                dev.positiveCorrelationCount + diag.positiveCorrelationCount
            )
            combined_significant = int(
                dev.positiveSignificantCount + diag.positiveSignificantCount
            )
            combined_higher = int(
                dev.higherDuringReplicationCount + diag.higherDuringReplicationCount
            )
            row = {
                "candidatePipelineId": spec.candidatePipelineId,
                "pipelineId": spec.pipelineId,
                "candidateId": candidate_id,
                "developmentDefined": int(dev.definedTrajectoryCount),
                "developmentPositiveFraction": dev.positiveCorrelationFraction,
                "developmentPositiveSignificantFraction": dev.positiveSignificantFraction,
                "developmentMedianCorrelation": dev.medianCorrelation,
                "developmentHigherDuringReplicationFraction": dev.higherDuringReplicationFraction,
                "developmentAggregateTrendP": dev.aggregateTrendP,
                "diagnosticDefined": int(diag.definedTrajectoryCount),
                "diagnosticPositiveFraction": diag.positiveCorrelationFraction,
                "diagnosticPositiveSignificantFraction": diag.positiveSignificantFraction,
                "diagnosticMedianCorrelation": diag.medianCorrelation,
                "diagnosticHigherDuringReplicationFraction": diag.higherDuringReplicationFraction,
                "diagnosticMedianMeanDifference": diag.medianMeanDifference,
                "diagnosticPositiveThreeSigmaRunFraction": diag.positiveThreeSigmaRunFraction,
                "diagnosticRobustSpikeRunFraction": diag.robustSpikeRunFraction,
                "diagnosticRawLjungBoxFraction": diag.rawLjungBoxFraction,
                "diagnosticDifferencedLjungBoxFraction": diag.differencedLjungBoxFraction,
                "diagnosticAggregateTrendP": diag.aggregateTrendP,
                "combinedDefined": combined_defined,
                "combinedPositiveCount": combined_positive,
                "combinedPositiveFraction": combined_positive / combined_defined,
                "combinedPositiveSignificantCount": combined_significant,
                "combinedPositiveSignificantFraction": combined_significant
                / combined_defined,
                "combinedHigherDuringReplicationCount": combined_higher,
                "combinedHigherDuringReplicationFraction": combined_higher
                / combined_defined,
            }
            if not infer.empty:
                observed = infer.iloc[0]
                row.update(
                    {
                        "diagnosticBootstrapLower95": observed.correlationBootstrapLower95,
                        "diagnosticBootstrapUpper95": observed.correlationBootstrapUpper95,
                        "diagnosticCircularShiftPositiveP": observed.circularShiftPositiveP,
                    }
                )
            else:
                row.update(
                    {
                        "diagnosticBootstrapLower95": None,
                        "diagnosticBootstrapUpper95": None,
                        "diagnosticCircularShiftPositiveP": None,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def gap_outcomes() -> pd.DataFrame:
    rows = [
        (
            1,
            "replicator label observation scope",
            "HIGH_LEVERAGE_PARTIAL_EXPLANATION",
            "Molecular incoming-similarity labels turn the regularized full-fit emergence association positive; all post-fission recurring/adjacent families remain near zero or negative.",
        ),
        (
            2,
            "level versus change association",
            "BOTH_CAN_RESEMBLE_DIRECTION",
            "PhiRL emergence levels and downward-causation changes both yielded positive adaptive branches; the level branch is closer to the Results prose.",
        ),
        (
            3,
            "temporal label placement",
            "HIGH_LEVERAGE",
            "Same-state molecular incoming labels are positive; nearby shifts and average labels vary, exposing an indexing dependency.",
        ),
        (
            4,
            "metric and atom identity",
            "PARTIALLY_RESOLVED",
            "Source-defined emergence works retrospectively under PhiRL; downward causation alone can score slightly higher but is not the paper's full scalar.",
        ),
        (
            5,
            "preprocessing and regularization",
            "REGULARIZATION_DEPENDENT",
            "The strongest results use PhiRL; the IIGR emergence comparator is weak/inconsistent.",
        ),
        (
            6,
            "partition identity",
            "UNRESOLVED",
            "S13X retained the source Fiedler partition; no favorable alternative partition was introduced.",
        ),
        (
            7,
            "local fit and window",
            "FUTURE_DEPENDENCE_CONFIRMED",
            "Completed-fit positive associations become negative under exact past-only prefix refits.",
        ),
        (
            8,
            "GARD dynamics and time base",
            "ROBUST_ACROSS_TWO_CONFIRMED_CANDIDATES",
            "The retrospective direction appears under both frozen confirmed time-base candidates, but this does not identify author semantics.",
        ),
        (
            9,
            "spike definition",
            "DIRECTIONALLY_RESEMBLES",
            "Most diagnostic runs show positive 3-sigma excursions and all show robust excursions; temporal fractions are not exact paper matches.",
        ),
        (
            10,
            "intervention scoring",
            "NOT_REPRODUCED",
            "Only 1/8 primary persistence/occupancy comparisons followed max >= control >= min in the four-triplet frozen-control pilot.",
        ),
    ]
    return pd.DataFrame(rows, columns=["rank", "gap", "resolution", "S13XFinding"])


def representative_trace_figure(specifications: pd.DataFrame) -> None:
    pipeline_id = specifications[
        specifications["candidatePipelineId"] == "S13X-C3-TABLE-OCCUPANCY-H097"
    ].iloc[0]["pipelineId"]
    details = pd.read_parquet(
        STEP_ROOT / "focused_diagnostic_trajectory_results.parquet"
    )
    details = details[details["pipelineId"] == pipeline_id]
    source = pd.read_parquet(SOURCE_VALUES)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for axis, candidate_id in zip(
        axes, ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"), strict=True
    ):
        candidate_details = details[details["candidateId"] == candidate_id]
        target = float(candidate_details["rho"].median())
        representative = candidate_details.iloc[
            (candidate_details["rho"] - target).abs().argsort().iloc[0]
        ]
        matrix_index = int(representative["matrixIndex"])
        values = source[
            (source["candidateId"] == candidate_id)
            & (source["matrixIndex"] == matrix_index)
            & (source["implementationId"] == "PHIRL_REGULARIZED_SOURCE")
        ].sort_values("selectedSequenceIndex")
        labels = pd.read_parquet(
            Path("/cache/e01_s13x_v1/labels")
            / candidate_id
            / f"M{matrix_index:03d}.parquet"
        )
        labels = labels[labels["labelId"] == "MOL_ADJACENT_INCOMING_H970"][
            ["selectedSequenceIndex", "isReplicator"]
        ]
        merged = values.merge(labels, on="selectedSequenceIndex", how="left")
        axis.plot(
            merged["selectedSequenceIndex"],
            merged["emergence"],
            linewidth=0.8,
            color="#2455a4",
        )
        replicated = merged[merged["isReplicator"].astype(bool)]
        axis.scatter(
            replicated["selectedSequenceIndex"],
            replicated["emergence"],
            s=4,
            alpha=0.3,
            color="#d94841",
            label="molecular H>0.97 label",
        )
        axis.set_title(
            f"{candidate_id}, diagnostic M{matrix_index:03d}, rho={representative.rho:.3f}"
        )
        axis.set_ylabel("PhiRL source emergence")
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("locked molecular observation index")
    fig.suptitle(
        "Representative retrospective full-fit punctuated trajectories (adaptively selected)"
    )
    fig.savefig(
        FIGURE_ROOT / "01_representative_retrospective_trajectories.png", dpi=180
    )
    plt.close(fig)


def summary_figures(results: pd.DataFrame) -> None:
    subset = results[
        results["candidatePipelineId"].isin(
            [
                "S13X-C1-PAPER-H090-MOLECULAR",
                "S13X-C2-TABLE-OCCUPANCY-H095",
                "S13X-C3-TABLE-OCCUPANCY-H097",
            ]
        )
    ].copy()
    names = {
        "S13X-C1-PAPER-H090-MOLECULAR": "H>.90",
        "S13X-C2-TABLE-OCCUPANCY-H095": "H>.95",
        "S13X-C3-TABLE-OCCUPANCY-H097": "H>.97",
    }
    subset["threshold"] = subset["candidatePipelineId"].map(names)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for candidate_id, group in subset.groupby("candidateId", sort=True):
        group = group.set_index("threshold").loc[["H>.90", "H>.95", "H>.97"]]
        axes[0].plot(
            group.index,
            group["combinedPositiveFraction"],
            marker="o",
            label=candidate_id,
        )
        axes[1].plot(
            group.index,
            group["combinedHigherDuringReplicationFraction"],
            marker="o",
            label=candidate_id,
        )
    axes[0].axhline(0.73, color="black", linestyle="--", label="paper 73/100")
    axes[1].axhline(0.57, color="black", linestyle="--", label="paper 57/100")
    axes[0].set_title("Positive emergence–label correlations")
    axes[1].set_title("Higher emergence during replication")
    for axis in axes:
        axis.set_ylim(0, 1)
        axis.set_ylabel("run fraction")
        axis.legend(fontsize=7)
    fig.suptitle("Adaptive retrospective directional resemblance across thresholds")
    fig.savefig(FIGURE_ROOT / "02_directional_resemblance.png", dpi=180)
    plt.close(fig)

    prefix = pd.read_csv(STEP_ROOT / "prefix_audit_results.csv")
    prefix = prefix[prefix["alignment"] == "CURRENT_ENDPOINT"]
    rows = []
    for item in subset.itertuples(index=False):
        label_id = {
            "H>.90": "MOL_ADJACENT_INCOMING_H900",
            "H>.95": "MOL_ADJACENT_INCOMING_H950",
            "H>.97": "MOL_ADJACENT_INCOMING_H970",
        }[item.threshold]
        past = prefix[
            (prefix["candidateId"] == item.candidateId)
            & (prefix["labelId"] == label_id)
        ].iloc[0]
        rows.extend(
            [
                {
                    "candidateId": item.candidateId,
                    "threshold": item.threshold,
                    "mode": "retrospective full",
                    "medianRho": item.diagnosticMedianCorrelation,
                },
                {
                    "candidateId": item.candidateId,
                    "threshold": item.threshold,
                    "mode": "past-only prefix",
                    "medianRho": past.medianCorrelation,
                },
            ]
        )
    comparison = pd.DataFrame(rows)
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4), sharey=True, constrained_layout=True
    )
    for axis, candidate_id in zip(
        axes, ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"), strict=True
    ):
        group = comparison[comparison["candidateId"] == candidate_id]
        x = np.arange(3)
        for offset, mode in ((-0.18, "retrospective full"), (0.18, "past-only prefix")):
            ordered = (
                group[group["mode"] == mode]
                .set_index("threshold")
                .loc[["H>.90", "H>.95", "H>.97"]]
            )
            axis.bar(x + offset, ordered["medianRho"], width=0.34, label=mode)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, ["H>.90", "H>.95", "H>.97"])
        axis.set_title(candidate_id)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("median within-run Spearman rho")
    fig.suptitle("Retrospective resemblance reverses under past-only source refitting")
    fig.savefig(FIGURE_ROOT / "03_retrospective_vs_prefix.png", dpi=180)
    plt.close(fig)

    intervention = pd.read_csv(STEP_ROOT / "intervention_directional_results.csv")
    intervention = intervention[
        (intervention["labelId"] == "MOL_ADJACENT_INCOMING_H970")
        & intervention["outcome"].isin(["persistence", "probability"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, outcome in zip(axes, ("persistence", "probability"), strict=True):
        group = intervention[intervention["outcome"] == outcome]
        for row in group.itertuples(index=False):
            label = f"{row.candidateId[-2:]} M{int(row.matrixIndex)}"
            axis.plot(
                ["MAX", "CONTROL", "MIN"],
                [row.max, row.control, row.min],
                marker="o",
                label=label,
            )
        axis.set_title(outcome)
        axis.legend(fontsize=7)
    fig.suptitle("Four-triplet retrospective frozen-control pilot")
    fig.savefig(FIGURE_ROOT / "04_intervention_pilot.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".4f")


def report_text(
    specs: pd.DataFrame,
    results: pd.DataFrame,
    gaps: pd.DataFrame,
    decision: dict[str, Any],
    prior: dict[str, Any],
    search: dict[str, Any],
) -> str:
    anchor = results[
        results["candidatePipelineId"].isin(
            [
                "S13X-C1-PAPER-H090-MOLECULAR",
                "S13X-C2-TABLE-OCCUPANCY-H095",
                "S13X-C3-TABLE-OCCUPANCY-H097",
            ]
        )
    ].copy()
    prefix = pd.read_csv(STEP_ROOT / "prefix_audit_inference.csv")
    prefix = prefix[
        (prefix["alignment"] == "CURRENT_ENDPOINT")
        & prefix["labelId"].isin(
            ["MOL_ADJACENT_INCOMING_H900", "MOL_ADJACENT_INCOMING_H970"]
        )
    ]
    intervention = pd.read_csv(STEP_ROOT / "intervention_directional_results.csv")
    intervention = intervention[
        (intervention["labelId"] == "MOL_ADJACENT_INCOMING_H970")
        & intervention["outcome"].isin(["persistence", "probability"])
    ]
    stability = pd.read_csv(STEP_ROOT / "focused_neighborhood_stability.csv")
    operational = pd.read_csv(STEP_ROOT / "operational_issue_ledger.csv")
    now = datetime.now(UTC).isoformat()
    return f"""# S13X full results — creative directional replication search

## Top summary

- **Research step ID:** `E01-S13X-CREATIVE-DIRECTIONAL-REPLICATION-SEARCH-v1.0.0` (`S13X`).
- **Completion status:** `COMPLETED`; only S13X was executed, and control returns here before E02 or report-bundle generation.
- **Artifacts written:** complete adaptive protocol, ranked gap inventory, {search["rowCount"]:,}-row chronological search ledger, candidate registry, development/diagnostic/prefix/intervention results, validation and provenance records, figures, status JSON, and hash manifest under `/artifacts/research_steps/S13X/`.
- **Validation result:** `PASS`. All 869 pre-S13X artifact hashes remained unchanged; all 206 frozen inputs matched; 200 label tasks and their sentinels replayed; source-prefix suffix evidence retained 3,552/3,552 executed passes; the four-triplet scorer replayed the source to {decision["maximumFixedScorerSourceReplayError"]:.3e}; all 12 pilot trajectories and actions replayed; seven focused tests and lint/compile checks passed.
- **Outcome classification:** `{OUTCOME_CLASSIFICATION}` — supportive exploratory retrospective resemblance, coupled to constraining/contradictory prospective and intervention evidence.
- **Caveats or blockers:** this was explicitly adaptive and outcome-guided; it searched thousands of specifications, reused the diagnostic split for focused follow-up, includes two deterministic duplicate executions in the ledger, and cannot identify the unavailable author implementation. The strongest relationship uses the later regularized PhiRL family and a molecular adjacent-similarity label, while completed-fit Gaussian/partition estimates use future data. Prefix values are unavailable for both matrix-72 tasks. The intervention pilot is tiny and retrospective.
- **Lay summary:** A paper-like observational direction can be reconstructed, but the key appears to be *where the replication label is applied*. Using the regularized public PhiRL emergence value on a completed trajectory and labelling every molecular state by similarity to its immediately previous state produces positive emergence–replication associations under both validated GARD clocks. With the paper's stated `H>0.9`, the combined positive count is 73/100 for candidate 2 and 79/99 for candidate 3. Raising the exploratory threshold to 0.97 makes occupancy resemble the paper and yields even stronger positive fractions. However, the same source code refit only on past data gives negative median correlations, and maximizing/minimizing the reconstructed score does not reliably move replication. This is a useful explanation of how the figures might arise retrospectively, not a recovered causal mechanism.
- **Recommended next action:** human review. If work continues, freeze the most source-directed `H>0.9` molecular-label/PhiRL-emergence branch *before* generating a genuinely new matrix set and test it alongside an explicit label-circularity control. Do not extend the intervention branch until scoring semantics are sourced. Do not begin E02 or report-bundle generation automatically.

## Frozen question and evidence posture

The human authorized a creative search whose sole aim was directional resemblance rather than satisfaction of prior gates. S13X therefore asked which missing simulator, label, information-theory, temporal, or intervention choices can reproduce: punctuated trajectories with weak aggregate trend, positive emergence–replication association, and higher emergence during replication. All results are labelled `ADAPTIVE_OUTCOME_GUIDED_EXPLORATORY_RECONSTRUCTION`. They are additive; S01–S13RRR and their classifications remain immutable.

The search began with two upstream-confirmed time-base simulators already represented by 100 paired matrices each: candidate 2 (`h=0.6031526490073492`, first daughter, trimmed overshoot, C1 clock) and candidate 3 (`h=0.5613315384859516`, random nonempty daughter, trimmed overshoot, C1 clock). This isolates downstream choices while checking robustness to the remaining simulator ambiguity.

## Lay interpretation

Three facts matter.

1. The original frozen post-fission replication labels occupy only about 30% of molecular observations and retain the prior near-zero/negative source-emergence result. The paper's prose can also be read as classifying composition at every molecular step. When S13X made that observation-scale change, the direction flipped.
2. The flip is not confined to a single threshold. It occurs at `H>0.9`, `H>0.95`, and `H>0.97`, although these thresholds trade off the paper's occupancy and other Table-1 fingerprints. The `H>0.97` label reaches about 0.87–0.89 occupancy on the diagnostic split, close to the reported 0.88, but its onset and consistency remain wrong.
3. The relationship is retrospective. Refitting the same PhiRL source pipeline on prefixes changes the sign, and a small controller using the completed control model fails the reported max/control/min direction. Thus S13X found a plausible *descriptive reconstruction gap*, not prospective early warning or causal control.

## Inputs and provenance

- Original paper: `/cache/e01_s03/downloads/paper-2607.28250v1.pdf`, with extracted text at `/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md`.
- Frozen full and prefix source outputs: `{SOURCE_VALUES}` and `{PREFIX_VALUES}`.
- Frozen 200 held-out raw trajectories: `/cache/e01_s13/raw_trajectories/S12F-CANDIDATE-02|03/M000..M099.pickle`.
- Safe PhiID lattice: `/artifacts/research_steps/S12B/safe_phi_lattice.json`; no pickle was loaded.
- Source identities: IIGR `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`; PhiRL `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; historical GARD `86dff6320d5ae91b4e831471079ff46749b14df9`.
- Repository branch/head at finalization: `{git("branch", "--show-current")}` / `{git("rev-parse", "HEAD")}`; remote head `{git("rev-parse", "origin/eidosoma/groups/42")}`.
- Prior immutability: {prior["fileCount"]} files checked, {prior["mismatchCount"]} mismatches.

## Methods

### Gap inventory and pre-outcome protocol

Before the systematic search, S13X ranked ten gaps: label observation scope, level-versus-change ambiguity, temporal label placement, atom identity, preprocessing/regularization, partitioning, completed-fit/window scope, GARD semantics, spike definition, and intervention scoring. The adaptive protocol, 206-input manifest, 869-file prior baseline, 22-label registry, and four pre-protocol discoveries were frozen at commit `6583610f9281930ea5e7a87528d158f765e1d8e0` before development outcomes were opened.

### Label search

The search materialized 22 explicit label specifications on every C1 observation of all 200 trajectories. Families included the historical post-fission adjacent `H>0.9`, Euclidean and recurring-centroid/medoid post-fission variants, and molecular incoming or incoming/outgoing-average cosine variants over a declared threshold grid. Development used matrices 0–59; matrices 60–99 were held aside for the first diagnostic pass. Applying `H>0.9` at the molecular clock was source-transplanted/paper-inferred; thresholds above 0.9 were explicitly Table-1-directed speculative sensitivities.

### Systematic information/temporal search

The first pass evaluated 7,744 unique combinations of two public implementations, four source scalar/atom identities, ten temporal transforms, 22 labels, and valid state/generation alignments. A continuous resemblance score used paper-reported directions rather than a success gate. Every attempt—including negative results—is present in `chronological_search_ledger.csv`. Twelve diversity-preserving candidates were evaluated on matrices 60–99 with 4,096 trajectory bootstraps and 4,096 within-trajectory circular-shift nulls.

After observing that first diagnostic result, the explicitly outcome-guided neighborhood evaluated 360 nearby combinations on the same diagnostic data. It was intentionally not treated as a second holdout. A final ten-branch paper-directed check contrasted the exact molecular `H>0.9` reading with post-fission recurring-composition definitions.

### Past-only audit

Existing S13RRR PhiRL prefix values were used without a new source fit. Each value had been independently fit on observations available through its post-fission endpoint after 256 transitions. Molecular incoming labels at these endpoints use only the previous and current state. Current- and next-endpoint alignments were kept separate. The S13RRR structural and 3,552 executed deletion/shuffle/replacement suffix checks were reused; candidate-2 and candidate-3 matrix 72 had no eligible endpoint and remained explicitly unavailable.

### Retrospective directional intervention pilot

Because a plausible source-defined branch emerged, S13X ran four max/control/min triplets: two new domain-separated matrices under each simulator, 12 trajectories total. The completed matched control fixed PhiRL retained variables, z-score moments, Fiedler partition, and Gaussian densities. At every fission, all 100 additions and every available deletion were scored as a virtual transition from the selected daughter to the edited state. Raw extrema were applied with additions-before-deletions/lowest-index deterministic ties. This is the S12E `I2`-like retrospective forensic semantics, not an online author implementation. The fixed scorer was validated against every control-source point before use.

## Results

### Candidate registry and anchor results

{markdown_table(specs, ["candidatePipelineId", "pipelineId", "implementationId", "metric", "transform", "labelId", "alignment", "role"])}

Combined development-plus-diagnostic counts are descriptive because the candidates were selected adaptively. Split-specific medians and the diagnostic resampling evidence remain visible in `directional_candidate_results.csv`.

{markdown_table(anchor, ["candidatePipelineId", "candidateId", "combinedDefined", "combinedPositiveCount", "combinedPositiveFraction", "combinedPositiveSignificantCount", "combinedPositiveSignificantFraction", "combinedHigherDuringReplicationCount", "combinedHigherDuringReplicationFraction", "diagnosticMedianCorrelation", "diagnosticBootstrapLower95", "diagnosticBootstrapUpper95", "diagnosticCircularShiftPositiveP"])}

The most source/paper-directed branch applies the stated `H>0.9` adjacent similarity on the molecular clock and uses PhiRL source-defined emergence levels. Candidate 2 gives exactly 73/100 positive correlations; candidate 3 gives 79/99. On the disjoint first diagnostic split alone, the respective positive fractions are 0.75 and 0.70, median correlations 0.0298 and 0.0466, and higher-during-replication fractions 0.825 and 0.800. Both diagnostic bootstrap intervals exclude zero and both circular-shift p-values are 1/4097.

At `H>0.97`, the diagnostic occupancy becomes 0.873 and 0.887, close to the paper's 0.88. Combined positive fractions become 0.90 and 0.86; positive-significant fractions 0.64 and 0.69; and higher-during-replication fractions 0.93 and 0.95. This is the strongest directional resemblance, but it is an outcome-guided threshold change, not recovery of the stated author threshold.

### What did not match

- Molecular `H>0.9` occupancy is about 0.98 rather than 0.88; `H>0.97` improves occupancy but leaves onset near 3 rather than 37 and consistency near 0.09 rather than 0.38.
- Post-fission historical, Euclidean, recurring-centroid, and medoid labels do not yield the positive association. Under PhiRL on the diagnostic split, the historical post-fission label has median rho -0.023/-0.026 and higher-during-replication fractions 0.243/0.378.
- The result is implementation dependent. The analogous IIGR `H>0.9` level branch is near zero; the best IIGR-emergence neighborhood branch is weak and inconsistent. This prevents treating PhiRL's later regularization as the paper's unidentified implementation.
- Punctuated structure resembles the paper but is not exact: for the molecular `H>0.9` branch, positive 3-sigma excursions occur in 82.5%/77.5% of diagnostic runs, robust excursions in 100%/100%, raw Ljung–Box significance in 77.5%/72.5%, and differenced significance in 100%/100%. Aggregate trend is weak for candidate 2 (`p=0.861`) but significant for candidate 3 (`p=0.032`) and was strongly significant in the development split.

Neighborhood stability by implementation and scalar:

{markdown_table(stability, list(stability.columns))}

### Prospective prefix audit

{markdown_table(prefix, ["candidateId", "labelId", "alignment", "definedCorrelationCount", "positiveCorrelationFraction", "medianCorrelation", "bootstrapLower95", "bootstrapUpper95", "circularShiftPositiveP", "higherDuringReplicationFraction", "medianMeanDifference"])}

Every current-endpoint molecular label reverses direction under past-only PhiRL refitting. For `H>0.9`, median rho is -0.062/-0.072; for the occupancy-matched `H>0.97`, -0.072/-0.071. Positive-run fractions range only 0.235–0.304. The reconstructed retrospective result is therefore future-fitting dependent and cannot support early warning, prediction, or causal action selection.

### Directional intervention pilot

{markdown_table(intervention, ["candidateId", "matrixIndex", "outcome", "max", "control", "min", "maxMinusControl", "controlMinusMin", "paperDirectedOrdering"])}

Only {decision["primaryPaperDirectedOrderingCount"]}/{decision["primaryDirectionalComparisonCount"]} primary persistence/occupancy comparisons followed the paper-directed ordering. All 12 trajectories reached 100 fissions, 97,061 candidate scores were retained, all trajectory/action/source replays passed, and the frozen scorer's maximum error against source emergence was {decision["maximumFixedScorerSourceReplayError"]:.3e}. The negative direction is thus scientific/semantic rather than a software failure for this scorer.

### Ranked gap disposition

{markdown_table(gaps, ["rank", "gap", "resolution", "S13XFinding"])}

## Validation

- Pre-S13X immutability: `{prior["passed"]}` ({prior["fileCount"]} files; {prior["mismatchCount"]} mismatches).
- Frozen input validation: `PASS` (206 inputs; zero mismatches).
- Label cache: 200 tasks, 22 specifications, 3,869,228 rows; eight exact sentinel replays.
- Stage-1 cardinality: 7,744 pipelines, 15,488 candidate-development summaries, 12 diagnostic pipelines, 960 diagnostic trajectory summaries.
- Focused neighborhood: 360 pipelines, 720 candidate summaries, 28,800 trajectory summaries. One accidental deterministic rerun is retained rather than hidden.
- Paper-directed check: 10 pipelines, 20 candidate summaries, 800 trajectory summaries.
- Anchor resampling completion: three already named pipelines, six candidate summaries, and 240 trajectory summaries; no new scientific specification.
- Prefix audit: 198 available trajectory tasks, 1,188 trajectory/specification summaries, 3,552/3,552 executed suffix sentinels; both matrix-72 tasks explicitly unavailable.
- Intervention: four triplets, 12/12 trajectories completed 100 fissions; exact replay passed for trajectories, action rankings, and source fits.
- Software: Ruff `PASS`; seven tests `PASS`; all S13X scripts compile.
- Chronology: {search["rowCount"]:,} ledger rows, monotonically unique sequence, {search["duplicateExecutionRowCount"]} rows marked as belonging to duplicate deterministic executions.
- Operational repairs: {len(operational)} transparent issues, all repaired or retained without changing scientific specifications.

## Commands

```bash
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13x_protocol.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13x_creative_directional_search.py --workers 6
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13x_focused_neighborhood.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13x_paper_directed_label_check.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13x_prefix_audit.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13x_directional_intervention_pilot.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13x_anchor_inference.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/finalize_s13x.py
PYTHONPATH=src python -m pytest -q tests/e01/test_s13x_creative_directional_search.py tests/e01/test_s13x_intervention.py
python -m ruff check <S13X source, scripts, and tests>
```

Every numerical-library thread variable was set to one for production runs. Stage 1 used six label workers; the intervention pilot used four independent triplet workers. CPU float64 was authoritative; the L4 was not used.

## Dependencies and environment

- Python `{platform.python_version()}` on `{platform.platform()}`.
- NumPy `{np.__version__}`, pandas `{pd.__version__}`, SciPy/statsmodels/scikit-learn/networkx/pyarrow/matplotlib from the supplied scientific environment.
- No new dependency was installed for S13X.

## Caveats, failed assumptions, and claim boundary

S13X is a specification search, not a confirmatory experiment. Its continuous resemblance score, threshold neighborhood, diagnostic reuse, and intervention trigger were outcome guided and must not be assigned a nominal familywise error rate. Strong within-run p-values do not correct specification search multiplicity. The molecular label is closely related to one-step compositional stability, so association with a local Gaussian information statistic may partly reflect a shared reaction coordinate or definitional coupling. Completed-fit partitions and Gaussian parameters use future data. PhiRL postdates/regularizes IIGR and is not proven to be the paper's implementation. The pilot controller is retrospective and tiny. None of this establishes exact author identity, fixed-window Phi-r, early warning, prediction, intervention efficacy, or causal emergence.

Negative evidence remains active: S10/S11/S11R estimator constraints, S12 strict negative associations and action suppression, S12C source-family non-support for corrected local Phi-r, S12D/S12E failures, S13RRR held-out non-support under the frozen historical labels, the new prefix reversal, and the new intervention non-reproduction.

## Provenance and artifact map

Key machine-readable files are `chronological_search_ledger.csv`, `candidate_registry.csv`, `directional_candidate_results.csv`, `gap_inventory_outcomes.csv`, `prefix_audit_results.csv`, `intervention_directional_results.csv`, `validation_summary.json`, `provenance_manifest.json`, `status.json`, and `artifact_manifest.json`. Bulky label and intervention caches remain under `/cache/e01_s13x_v1/`; only compact evidence and cache manifests are collectible.

Finalized at `{now}`. S13X stops here for human review. No E02 or report-bundle work was started.
"""


def make_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(STEP_ROOT.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(STEP_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": "eidosoma.e01.s13x_artifact_manifest.v1",
        "researchStepId": "S13X",
        "versionedStepId": VERSION,
        "fileCountExcludingManifest": len(files),
        "totalBytesExcludingManifest": sum(item["bytes"] for item in files),
        "files": files,
    }
    write_json(STEP_ROOT / "artifact_manifest.json", manifest)
    return manifest


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    prior = prior_immutability()
    search = normalize_search_ledger()
    registry = pd.read_csv(STEP_ROOT / "pipeline_search_registry.csv")
    specs = candidate_specifications(registry)
    results = candidate_results(specs)
    gaps = gap_outcomes()
    specs.to_csv(STEP_ROOT / "candidate_registry.csv", index=False, lineterminator="\n")
    results.to_csv(
        STEP_ROOT / "directional_candidate_results.csv",
        index=False,
        lineterminator="\n",
    )
    gaps.to_csv(
        STEP_ROOT / "gap_inventory_outcomes.csv", index=False, lineterminator="\n"
    )

    intervention_validation = json.loads(
        (STEP_ROOT / "intervention_pilot_validation.json").read_text(encoding="utf-8")
    )
    prefix = pd.read_csv(STEP_ROOT / "prefix_audit_results.csv")
    decision = {
        "schema": "eidosoma.e01.s13x_decision.v1",
        "researchStepId": "S13X",
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "classification": OUTCOME_CLASSIFICATION,
        "outcomeClass": "SUPPORTIVE_EXPLORATORY_WITH_CONTRADICTORY_CAUSALIZATION",
        "retrospectiveDirectionalResemblanceFound": True,
        "mostSourceDirectedPipelineId": str(
            specs[specs["candidatePipelineId"] == "S13X-C1-PAPER-H090-MOLECULAR"].iloc[
                0
            ]["pipelineId"]
        ),
        "tableOccupancyMatchedPipelineId": str(
            specs[specs["candidatePipelineId"] == "S13X-C3-TABLE-OCCUPANCY-H097"].iloc[
                0
            ]["pipelineId"]
        ),
        "prospectivePrefixSupportFound": bool((prefix["medianCorrelation"] > 0).all()),
        "interventionDirectionalMatchFound": bool(
            intervention_validation["primaryPaperDirectedOrderingCount"]
            == intervention_validation["primaryDirectionalComparisonCount"]
        ),
        "regularizationDependent": True,
        "exactAuthorIdentityClaimed": False,
        "causalProofClaimed": False,
        "E02Started": False,
        "reportBundleStarted": False,
        **{
            key: intervention_validation[key]
            for key in (
                "primaryPaperDirectedOrderingCount",
                "primaryDirectionalComparisonCount",
                "maximumFixedScorerSourceReplayError",
            )
        },
        "recommendedNextAction": (
            "Human review; if authorized, preregister a genuinely new-matrix validation "
            "of the fixed molecular-H>0.9/PhiRL-emergence branch with label-circularity "
            "controls. Do not continue the intervention branch without source evidence."
        ),
    }
    write_json(STEP_ROOT / "decision.json", decision)

    representative_trace_figure(specs)
    summary_figures(results)

    failures = pd.DataFrame(
        [
            (
                "S13X-F001",
                "operational",
                "REPAIRED",
                "focused neighborhood import path",
                "No scientific choice changed.",
            ),
            (
                "S13X-F002",
                "operational",
                "RETAINED",
                "unintended deterministic focused rerun",
                "Both ledger occurrences retained and marked.",
            ),
            (
                "S13X-F003",
                "operational",
                "REPAIRED",
                "prefix expected 200 rather than available 198 tasks",
                "Matrix-72 unavailability remains explicit.",
            ),
            (
                "S13X-C001",
                "scientific",
                "CONSTRAINING",
                "IIGR emergence does not reproduce the PhiRL lead",
                "Result is regularization/source-version dependent.",
            ),
            (
                "S13X-C002",
                "scientific",
                "CONTRADICTORY",
                "past-only prefix associations are negative",
                "No prospective early-warning support.",
            ),
            (
                "S13X-C003",
                "scientific",
                "CONTRADICTORY",
                "only 1/8 pilot comparisons has paper-directed ordering",
                "No intervention-direction support for the tested scorer.",
            ),
            (
                "S13X-C004",
                "scientific",
                "UNRESOLVED",
                "label threshold, observation scope, and author implementation remain unidentified",
                "No exact replication or author-primary claim.",
            ),
        ],
        columns=["failureId", "type", "status", "description", "impact"],
    )
    failures.to_csv(STEP_ROOT / "failure_ledger.csv", index=False, lineterminator="\n")

    stage_runtime = json.loads(
        (STEP_ROOT / "stage1_runtime.json").read_text(encoding="utf-8")
    )
    runtime = {
        "schema": "eidosoma.e01.s13x_runtime_manifest.v1",
        "researchStepId": "S13X",
        "stage1WallSeconds": stage_runtime["totalWallSeconds"],
        "stage1LabelSeconds": stage_runtime["labelSeconds"],
        "focusedNeighborhoodObservedWallSecondsApproximate": 103.1,
        "paperDirectedCheckObservedWallSecondsApproximate": 71.7,
        "interventionPilotWallSeconds": intervention_validation["wallSeconds"],
        "workers": {"labels": 6, "interventionTriplets": 4},
        "blasThreadsPerWorker": 1,
        "cpuPrecision": "float64_authoritative",
        "gpuUsed": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "retainedArtifactBytesBeforeManifest": sum(
            path.stat().st_size for path in STEP_ROOT.rglob("*") if path.is_file()
        ),
        "hardPlatformLimitsRespected": True,
    }
    write_json(STEP_ROOT / "runtime_manifest.json", runtime)
    provenance = {
        "schema": "eidosoma.e01.s13x_provenance_manifest.v1",
        "researchStepId": "S13X",
        "versionedStepId": VERSION,
        "evidenceClass": EVIDENCE_CLASS,
        "repositoryBranch": git("branch", "--show-current"),
        "repositoryHead": git("rev-parse", "HEAD"),
        "repositoryRemoteHead": git("rev-parse", "origin/eidosoma/groups/42"),
        "protocolCommit": "6583610f9281930ea5e7a87528d158f765e1d8e0",
        "sourceCommits": {
            "historicalGARD": "86dff6320d5ae91b4e831471079ff46749b14df9",
            "IIGR": "7c1c22fe39f539d4a453135476f1f0dd5a6b45f7",
            "PhiRL": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373",
        },
        "safeLatticeSha256": sha256_file(
            ARTIFACTS / "research_steps/S12B/safe_phi_lattice.json"
        ),
        "fullSourceValues": {
            "path": str(SOURCE_VALUES),
            "sha256": sha256_file(SOURCE_VALUES),
        },
        "prefixValues": {
            "path": str(PREFIX_VALUES),
            "sha256": sha256_file(PREFIX_VALUES),
        },
        "rawTrajectoryRoot": "/cache/e01_s13/raw_trajectories",
        "derivedCacheRoot": "/cache/e01_s13x_v1",
        "newPilotTrajectoryCount": 12,
        "priorArtifactsMutable": False,
        "priorImmutabilityPassed": prior["passed"],
    }
    write_json(STEP_ROOT / "provenance_manifest.json", provenance)

    validations = {
        name: json.loads((STEP_ROOT / name).read_text(encoding="utf-8"))
        for name in (
            "protocol_validation.json",
            "input_validation.json",
            "label_replay_validation.json",
            "stage1_validation.json",
            "focused_neighborhood_validation.json",
            "paper_directed_label_validation.json",
            "anchor_inference_validation.json",
            "prefix_audit_validation.json",
            "intervention_pilot_validation.json",
            "software_validation.json",
            "prior_immutability_validation.json",
        )
    }
    validation_summary = {
        "schema": "eidosoma.e01.s13x_validation_summary.v1",
        "researchStepId": "S13X",
        "checks": {
            name: bool(value.get("passed")) for name, value in validations.items()
        },
        "searchLedger": search,
        "allChecksPassed": all(
            bool(value.get("passed")) for value in validations.values()
        )
        and search["attemptSequenceUnique"]
        and search["attemptSequenceMonotonic"],
    }
    write_json(STEP_ROOT / "validation_summary.json", validation_summary)
    if not validation_summary["allChecksPassed"]:
        raise RuntimeError("S13X final validation summary failed")

    report = report_text(specs, results, gaps, decision, prior, search)
    (STEP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": "S13X",
        "stepNumber": "S13X",
        "success": True,
        "status": "COMPLETED",
        "artifactsWritten": [
            "/artifacts/research_steps/S13X/research_step_full_results.md",
            "/artifacts/research_steps/S13X/chronological_search_ledger.csv",
            "/artifacts/research_steps/S13X/candidate_registry.csv",
            "/artifacts/research_steps/S13X/directional_candidate_results.csv",
            "/artifacts/research_steps/S13X/prefix_audit_results.csv",
            "/artifacts/research_steps/S13X/intervention_directional_results.csv",
            "/artifacts/research_steps/S13X/artifact_manifest.json",
        ],
        "validationResult": "PASS",
        "outcomeClassification": OUTCOME_CLASSIFICATION,
        "caveatsOrBlockers": [
            "Adaptive outcome-guided search; not confirmatory.",
            "Retrospective lead depends on PhiRL regularization, molecular label scope, and completed-fit parameters.",
            "Past-only associations are negative and the intervention pilot does not reproduce the paper direction.",
            "Unavailable author implementation and exact label/partition/window semantics remain unresolved.",
        ],
        "recommendedNextAction": decision["recommendedNextAction"],
    }
    write_json(STEP_ROOT / "status.json", status)
    make_manifest()
    required = [
        "research_step_full_results.md",
        "status.json",
        "artifact_manifest.json",
        "candidate_registry.csv",
        "chronological_search_ledger.csv",
        "directional_candidate_results.csv",
        "prefix_audit_results.csv",
        "intervention_directional_results.csv",
        "validation_summary.json",
        "provenance_manifest.json",
        "failure_ledger.csv",
        "figures/01_representative_retrospective_trajectories.png",
        "figures/02_directional_resemblance.png",
        "figures/03_retrospective_vs_prefix.png",
        "figures/04_intervention_pilot.png",
    ]
    completeness = {
        "schema": "eidosoma.e01.s13x_artifact_completeness.v1",
        "researchStepId": "S13X",
        "required": required,
        "missing": [name for name in required if not (STEP_ROOT / name).is_file()],
    }
    completeness["passed"] = not completeness["missing"]
    write_json(STEP_ROOT / "artifact_completeness_validation.json", completeness)
    make_manifest()
    if not completeness["passed"]:
        raise RuntimeError("S13X artifact completeness failed")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
