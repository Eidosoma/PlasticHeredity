#!/usr/bin/env python3
"""Complete frozen resampling outputs for the three named S13X anchors."""

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
    evaluate,
    label_fingerprint_summary,
    pipeline_registry,
    write_csv,
    write_json,
)

ANCHOR_LABELS = (
    "MOL_ADJACENT_INCOMING_H900",
    "MOL_ADJACENT_INCOMING_H950",
    "MOL_ADJACENT_INCOMING_H970",
)


def main() -> None:
    source = pd.read_parquet(SOURCE_VALUES)
    fingerprints = pd.read_parquet(STEP_ROOT / "label_fingerprints.parquet")
    summary = label_fingerprint_summary(fingerprints)
    registry = pipeline_registry()
    anchors = registry[
        (registry["implementationId"] == "PHIRL_REGULARIZED_SOURCE")
        & (registry["metric"] == "emergence")
        & (registry["transform"] == "LEVEL")
        & registry["labelId"].isin(ANCHOR_LABELS)
        & (registry["alignment"] == "SAME_STATE")
    ]
    if len(anchors) != 3:
        raise RuntimeError("S13X anchor registry cardinality changed")
    results, details, payloads = evaluate(
        source,
        registry,
        summary,
        phase="DIAGNOSTIC",
        pipeline_ids=set(anchors["pipelineId"]),
        include_details=True,
    )
    inference = diagnostic_inference(results, details, payloads)
    write_csv(STEP_ROOT / "anchor_diagnostic_inference.csv", inference)
    validation = {
        "schema": "eidosoma.e01.s13x_anchor_inference_validation.v1",
        "researchStepId": "S13X",
        "anchorPipelineCount": len(anchors),
        "candidateResultCount": len(results),
        "trajectoryResultCount": len(details),
        "inferenceResultCount": len(inference),
        "newScientificSpecificationIntroduced": False,
        "passed": bool(
            len(results) == 6
            and len(details) == 240
            and len(inference) == 6
            and inference["status"].eq("VALID").all()
        ),
    }
    write_json(STEP_ROOT / "anchor_inference_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError("S13X anchor inference validation failed")
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
