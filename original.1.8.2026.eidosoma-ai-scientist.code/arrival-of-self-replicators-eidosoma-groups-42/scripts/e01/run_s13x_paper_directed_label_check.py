#!/usr/bin/env python3
"""Evaluate the paper/source-directed label subset after the adaptive lead."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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

LABELS = (
    "MOL_ADJACENT_INCOMING_H900",
    "PF_HISTORICAL_ADJACENT_AVERAGE_H090",
    "PF_EUCLIDEAN_KMEANS_DOMINANT",
    "PF_DOMINANT_COMPONENT_CENTROID_H900",
    "PF_MAX_NEIGHBOR_MEDOID_H900",
)


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
                "attemptId": f"S13X-PAPER-LABEL-A{offset:02d}",
                "phase": "POST_LEAD_PAPER_DIRECTED_LABEL_CHECK",
                "choiceFamily": "LABEL_OBSERVATION_SCOPE",
                "specification": json.dumps(
                    {
                        "implementationId": row.implementationId,
                        "metric": "emergence",
                        "transform": "LEVEL",
                        "labelId": row.labelId,
                        "alignment": "SAME_STATE",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "evidenceTier": f"ADAPTIVE_PAPER_DIRECTED/{row.labelEvidenceTier}",
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
                    "Checks whether molecular-clock placement rather than threshold/"
                    "recurrence wording is the active reconstruction gap; exploratory only."
                ),
            }
        )
    write_csv(path, pd.concat([existing, pd.DataFrame(rows)], ignore_index=True))


def main() -> None:
    source = pd.read_parquet(SOURCE_VALUES)
    fingerprints = pd.read_parquet(STEP_ROOT / "label_fingerprints.parquet")
    summary = label_fingerprint_summary(fingerprints)
    complete_registry = pipeline_registry()
    registry = complete_registry[
        complete_registry["implementationId"].isin(
            ["IIGR_CORRECTED_SOURCE", "PHIRL_REGULARIZED_SOURCE"]
        )
        & (complete_registry["metric"] == "emergence")
        & (complete_registry["transform"] == "LEVEL")
        & complete_registry["labelId"].isin(LABELS)
        & (complete_registry["alignment"] == "SAME_STATE")
    ].copy()
    if len(registry) != 10:
        raise RuntimeError(
            f"paper-directed registry cardinality changed: {len(registry)}"
        )
    results, details, payloads = evaluate(
        source,
        complete_registry,
        summary,
        phase="DIAGNOSTIC",
        pipeline_ids=set(registry["pipelineId"]),
        include_details=True,
    )
    ranking = ensemble_ranking(results)
    inference = diagnostic_inference(results, details, payloads)
    write_csv(STEP_ROOT / "paper_directed_label_registry.csv", registry)
    write_csv(STEP_ROOT / "paper_directed_label_results.csv", results)
    write_parquet(
        STEP_ROOT / "paper_directed_label_trajectory_results.parquet", details
    )
    write_csv(STEP_ROOT / "paper_directed_label_ranking.csv", ranking)
    write_csv(STEP_ROOT / "paper_directed_label_inference.csv", inference)
    append_ledger(ranking, registry)
    validation = {
        "schema": "eidosoma.e01.s13x_paper_directed_label_validation.v1",
        "researchStepId": "S13X",
        "adaptiveAfterLead": True,
        "pipelineCount": len(registry),
        "candidateResultCount": len(results),
        "trajectoryResultCount": len(details),
        "inferenceResultCount": len(inference),
        "passed": bool(
            len(registry) == 10
            and len(results) == 20
            and len(details) == 800
            and len(inference) == 20
        ),
    }
    write_json(STEP_ROOT / "paper_directed_label_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError("paper-directed label check validation failed")
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
