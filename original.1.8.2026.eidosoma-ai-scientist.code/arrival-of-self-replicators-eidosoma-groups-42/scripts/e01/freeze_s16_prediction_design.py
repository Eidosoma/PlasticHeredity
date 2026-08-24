#!/usr/bin/env python3
"""Materialize the outcome-blind S16 split and tensor/model manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

from e01_prediction_reconstruction.core import (
    EXPECTED_PARAMETER_COUNT,
    FEATURE_IDS,
    VERSION,
    MaskedSequenceMLP,
    build_split_manifest,
    parameter_count,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/e01/s16_first_quarter_prediction_preregistration.yaml"
SPLIT_PATH = REPO_ROOT / "configs/e01/s16_split_manifest.csv"
MANIFEST_PATH = REPO_ROOT / "configs/e01/s16_tensor_model_manifest.json"
LABEL_PATH = Path("/artifacts/research_steps/S13Y/label_values.parquet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["versionedStepId"] != VERSION:
        raise ValueError("S16 config/version mismatch")
    split = build_split_manifest()
    split.to_csv(SPLIT_PATH, index=False, lineterminator="\n")

    parquet = pq.ParquetFile(LABEL_PATH)
    label_schema = parquet.schema_arrow
    metadata = pd.read_parquet(
        LABEL_PATH,
        columns=["candidateId", "matrixIndex", "labelId", "selectedSequenceIndex"],
    )
    metadata = metadata.loc[metadata["labelId"].eq("MOL_ADJACENT_INCOMING_H900")]
    lengths = (
        metadata.groupby(["candidateId", "matrixIndex"])["selectedSequenceIndex"]
        .agg(["size", "min", "max"])
        .reset_index()
    )
    if len(lengths) != 200 or lengths["min"].ne(0).any():
        raise ValueError("frozen label identity/cardinality metadata mismatch")
    observed_maximum_t = int(lengths["size"].max())
    observed_maximum_input = int((lengths["size"] // 4).max())
    observed_maximum_target = int((lengths["size"] - lengths["size"] // 4).max())
    tensor = config["tensorLayout"]
    if (
        observed_maximum_t != tensor["maximumObservedT"]
        or observed_maximum_input != tensor["maximumInputLength"]
        or observed_maximum_target != tensor["maximumTargetLength"]
    ):
        raise ValueError("frozen tensor dimensions do not cover schema-only lengths")
    model = MaskedSequenceMLP().to(dtype=torch.float64)
    count = parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise ValueError("frozen architecture parameter count mismatch")
    payload = {
        "schema": "eidosoma.e01.s16_tensor_model_manifest.v1",
        "researchStepId": "S16",
        "versionedStepId": VERSION,
        "predictionOutcomeAccessed": False,
        "selectionBasis": "paper_text_plus_frozen_C1_schema_plus_pinned_PhiRL_source_conventions",
        "paperSpecificationAudit": config["paperAndSchemaResolution"],
        "temporalModes": config["temporalModes"],
        "tensorLayout": tensor,
        "featureFamilies": config["featureFamilies"],
        "scaling": config["scaling"],
        "splitsAndSeeds": config["splitsAndSeeds"],
        "model": config["model"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "classification": config["classification"],
        "observedSchemaOnly": {
            "primaryLabelTableRows": parquet.metadata.num_rows,
            "primaryCandidateMatrixUnits": len(lengths),
            "maximumT": observed_maximum_t,
            "maximumInputLength": observed_maximum_input,
            "maximumTargetLength": observed_maximum_target,
            "labelSchema": str(label_schema),
            "featureIds": list(FEATURE_IDS),
            "trainableParameterCount": count,
        },
        "files": {
            "preregistration": {
                "path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
                "sha256": sha256(CONFIG_PATH),
            },
            "splitManifest": {
                "path": str(SPLIT_PATH.relative_to(REPO_ROOT)),
                "sha256": sha256(SPLIT_PATH),
                "rows": len(split),
            },
        },
        "forbiddenAfterOutcomeAccess": config["forbiddenAfterOutcomeAccess"],
    }
    MANIFEST_PATH.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(
        canonical_json(
            {
                "predictionOutcomeAccessed": False,
                "splitRows": len(split),
                "parameterCount": count,
                "tensorManifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
                "tensorManifestSha256": sha256(MANIFEST_PATH),
            }
        )
    )


if __name__ == "__main__":
    main()
