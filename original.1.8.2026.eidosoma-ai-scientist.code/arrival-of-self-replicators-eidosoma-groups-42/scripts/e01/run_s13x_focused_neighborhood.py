#!/usr/bin/env python3
"""Run the adaptive S13X neighborhood check selected after stage-1 outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from e01_creative_directional_search.core import CANDIDATE_IDS
from scripts.e01.run_s13x_creative_directional_search import (
    SOURCE_VALUES,
    STEP_ROOT,
    diagnostic_inference,
    ensemble_ranking,
    evaluate,
    label_fingerprint_summary,
    pipeline_registry,
    write_csv,
    write_json,
    write_parquet,
)


def focused_registry(registry: pd.DataFrame) -> pd.DataFrame:
    """Return the outcome-guided but fully enumerated local neighborhood."""

    label_ids = {
        f"MOL_ADJACENT_{direction}_H{threshold}"
        for direction in ("INCOMING", "AVERAGE")
        for threshold in ("950", "965", "970", "975", "980")
    }
    result = registry[
        registry["implementationId"].isin(
            ["IIGR_CORRECTED_SOURCE", "PHIRL_REGULARIZED_SOURCE"]
        )
        & registry["metric"].isin(["emergence", "downwardCausation"])
        & registry["transform"].isin(
            ["LEVEL", "BACKWARD_DIFFERENCE", "FORWARD_DIFFERENCE"]
        )
        & registry["labelId"].isin(label_ids)
        & registry["alignment"].isin(["SAME_STATE", "NEXT_STATE", "PREVIOUS_STATE"])
    ].copy()
    result.sort_values("searchSequence", kind="stable", inplace=True, ignore_index=True)
    if len(result) != 360:
        raise RuntimeError(f"focused neighborhood cardinality changed: {len(result)}")
    return result


def inference_selection(ranking: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Keep the top neighborhood plus fixed scientific-family representatives."""

    merged = ranking.merge(registry, on="pipelineId", how="left", validate="one_to_one")
    selected: list[pd.Series] = []
    seen: set[str] = set()

    def add(row: pd.Series) -> None:
        pipeline_id = str(row["pipelineId"])
        if pipeline_id not in seen:
            selected.append(row)
            seen.add(pipeline_id)

    for _, row in merged.head(12).iterrows():
        add(row)
    for implementation in ("IIGR_CORRECTED_SOURCE", "PHIRL_REGULARIZED_SOURCE"):
        for metric in ("emergence", "downwardCausation"):
            subset = merged[
                (merged["implementationId"] == implementation)
                & (merged["metric"] == metric)
            ]
            if not subset.empty:
                add(subset.iloc[0])
    result = pd.DataFrame(selected)
    result.insert(0, "inferenceSelectionOrder", range(1, len(result) + 1))
    return result


def append_ledger(ranking: pd.DataFrame, registry: pd.DataFrame) -> None:
    path = STEP_ROOT / "chronological_search_ledger.csv"
    existing = pd.read_csv(path)
    start = int(existing["attemptSequence"].max()) + 1
    merged = ranking.merge(registry, on="pipelineId", how="left", validate="one_to_one")
    rows = []
    for offset, row in enumerate(
        merged.sort_values("searchSequence", kind="stable").itertuples(index=False)
    ):
        rows.append(
            {
                "attemptSequence": start + offset,
                "attemptId": f"S13X-NEIGHBOR-A{offset:04d}",
                "phase": "POST_DIAGNOSTIC_OUTCOME_GUIDED_NEIGHBORHOOD",
                "choiceFamily": "SOURCE_FAMILY_METRIC_TEMPORAL_LABEL_NEIGHBORHOOD",
                "specification": json.dumps(
                    {
                        "implementationId": row.implementationId,
                        "metric": row.metric,
                        "transform": row.transform,
                        "labelId": row.labelId,
                        "alignment": row.alignment,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "evidenceTier": (
                    "ADAPTIVE_AFTER_STAGE1/"
                    f"{row.labelEvidenceTier}/{row.metricEvidenceTier}/"
                    f"{row.transformEvidenceTier}"
                ),
                "outcome": (
                    f"rank={int(row.developmentRank)}; "
                    f"ensembleScore={float(row.ensembleDirectionalScore):.6f}; "
                    f"bothMedianPositive={bool(row.bothMedianCorrelationsPositive)}; "
                    f"bothDriftMajority={bool(row.bothHigherDuringReplicationMajority)}"
                ),
                "negativeResult": bool(
                    not row.bothMedianCorrelationsPositive
                    or not row.bothHigherDuringReplicationMajority
                ),
                "selectionUse": (
                    "Outcome-guided neighborhood characterization only; no new holdout "
                    "and no confirmatory interpretation."
                ),
            }
        )
    write_csv(path, pd.concat([existing, pd.DataFrame(rows)], ignore_index=True))


def main() -> None:
    source = pd.read_parquet(SOURCE_VALUES)
    fingerprints = pd.read_parquet(STEP_ROOT / "label_fingerprints.parquet")
    summary = label_fingerprint_summary(fingerprints)
    registry = focused_registry(pipeline_registry())
    results, details, payloads = evaluate(
        source,
        pipeline_registry(),
        summary,
        phase="DIAGNOSTIC",
        pipeline_ids=set(registry["pipelineId"]),
        include_details=True,
    )
    ranking = ensemble_ranking(results)
    selected = inference_selection(ranking, registry)
    selected_ids = set(selected["pipelineId"])
    inference = diagnostic_inference(
        results[results["pipelineId"].isin(selected_ids)],
        details[details["pipelineId"].isin(selected_ids)],
        {key: value for key, value in payloads.items() if key[0] in selected_ids},
    )
    write_csv(STEP_ROOT / "focused_neighborhood_registry.csv", registry)
    write_csv(STEP_ROOT / "focused_diagnostic_results.csv", results)
    write_parquet(STEP_ROOT / "focused_diagnostic_trajectory_results.parquet", details)
    write_csv(STEP_ROOT / "focused_diagnostic_ranking.csv", ranking)
    write_csv(STEP_ROOT / "focused_inference_registry.csv", selected)
    write_csv(STEP_ROOT / "focused_diagnostic_inference.csv", inference)
    append_ledger(ranking, registry)

    joined = results.merge(
        registry[["pipelineId", "implementationId", "metric"]],
        on=["pipelineId", "implementationId", "metric"],
        how="left",
        validate="many_to_one",
    )
    ensemble = ranking.merge(
        registry[["pipelineId", "implementationId", "metric"]],
        on="pipelineId",
        how="left",
        validate="one_to_one",
    )
    stability_rows = []
    for (implementation, metric), group in ensemble.groupby(
        ["implementationId", "metric"], sort=True
    ):
        stability_rows.append(
            {
                "implementationId": implementation,
                "metric": metric,
                "pipelineCount": len(group),
                "bothMedianPositiveCount": int(
                    group["bothMedianCorrelationsPositive"].sum()
                ),
                "bothMedianPositiveFraction": float(
                    group["bothMedianCorrelationsPositive"].mean()
                ),
                "bothDriftMajorityCount": int(
                    group["bothHigherDuringReplicationMajority"].sum()
                ),
                "bothDriftMajorityFraction": float(
                    group["bothHigherDuringReplicationMajority"].mean()
                ),
                "bothDirectionsCount": int(
                    (
                        group["bothMedianCorrelationsPositive"]
                        & group["bothHigherDuringReplicationMajority"]
                    ).sum()
                ),
                "bothDirectionsFraction": float(
                    (
                        group["bothMedianCorrelationsPositive"]
                        & group["bothHigherDuringReplicationMajority"]
                    ).mean()
                ),
                "medianEnsembleDirectionalScore": float(
                    group["ensembleDirectionalScore"].median()
                ),
                "maximumEnsembleDirectionalScore": float(
                    group["ensembleDirectionalScore"].max()
                ),
            }
        )
    stability = pd.DataFrame(stability_rows)
    write_csv(STEP_ROOT / "focused_neighborhood_stability.csv", stability)
    payload = {
        "schema": "eidosoma.e01.s13x_focused_neighborhood_validation.v1",
        "researchStepId": "S13X",
        "adaptiveAfterStage1Outcome": True,
        "pipelineCount": len(registry),
        "candidateResultCount": len(results),
        "trajectoryResultCount": len(details),
        "inferencePipelineCount": len(selected),
        "inferenceResultCount": len(inference),
        "candidateIds": list(CANDIDATE_IDS),
        "sourceInput": str(SOURCE_VALUES),
        "topPipelineId": str(ranking.iloc[0]["pipelineId"]),
        "topEnsembleDirectionalScore": float(
            ranking.iloc[0]["ensembleDirectionalScore"]
        ),
        "joinedCandidateResultCount": len(joined),
        "passed": bool(
            len(registry) == 360
            and len(results) == 720
            and len(details) == 28_800
            and len(inference) == 2 * len(selected)
            and set(results["candidateId"]) == set(CANDIDATE_IDS)
        ),
    }
    write_json(STEP_ROOT / "focused_neighborhood_validation.json", payload)
    if not payload["passed"]:
        raise RuntimeError("focused S13X neighborhood validation failed")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
