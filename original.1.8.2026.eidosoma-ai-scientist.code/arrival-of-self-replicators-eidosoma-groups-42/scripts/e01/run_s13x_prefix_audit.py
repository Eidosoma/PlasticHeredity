#!/usr/bin/env python3
"""Audit whether the adaptive S13X retrospective lead survives prefix refits."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from e01_creative_directional_search.core import association_summary, derive_seed
from scripts.e01.run_s13x_creative_directional_search import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_IDS,
    LABEL_CACHE,
    SHIFT_REPLICATES,
    STEP_ROOT,
    _circular_vectors,
    write_csv,
    write_json,
    write_parquet,
)

PREFIX_INPUT = Path("/artifacts/research_steps/S13RRR/prefix_endpoint_values.parquet")
SUFFIX_INPUT = Path("/artifacts/research_steps/S13RRR/replay_suffix_validation.parquet")
LABEL_IDS = (
    "MOL_ADJACENT_INCOMING_H900",
    "MOL_ADJACENT_INCOMING_H950",
    "MOL_ADJACENT_INCOMING_H970",
)
ALIGNMENTS = ("CURRENT_ENDPOINT", "NEXT_ENDPOINT")


def label_cache(candidate_id: str, matrix_index: int, label_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(LABEL_CACHE / candidate_id / f"M{matrix_index:03d}.parquet")
    return frame[frame["labelId"] == label_id][
        ["selectedSequenceIndex", "generation", "isReplicator", "labelScore"]
    ].copy()


def causal_label_check(frame: pd.DataFrame, label_id: str) -> bool:
    threshold = float(label_id.rsplit("H", 1)[1]) / 1000.0
    values = pd.to_numeric(frame["labelScore"], errors="coerce").to_numpy(float)
    expected = values > threshold
    observed = frame["isReplicator"].astype(bool).to_numpy()
    return bool(np.array_equal(expected, observed) and np.all(np.isfinite(values)))


def build_details(
    prefix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[Any, ...], Any]]:
    rows = []
    payloads = {}
    for candidate_id in CANDIDATE_IDS:
        candidate = prefix[prefix["candidateId"] == candidate_id]
        for matrix_index, group in candidate.groupby("matrixIndex", sort=True):
            ordered = group.sort_values("generation", kind="stable").copy()
            for label_id in LABEL_IDS:
                labels = label_cache(candidate_id, int(matrix_index), label_id)
                if not causal_label_check(labels, label_id):
                    raise RuntimeError(
                        f"past-only incoming-label identity failed: {label_id}"
                    )
                merged = ordered.merge(
                    labels,
                    left_on="endpointSelectedSequenceIndex",
                    right_on="selectedSequenceIndex",
                    how="left",
                    validate="many_to_one",
                    suffixes=("", "Label"),
                )
                if merged["isReplicator"].isna().any():
                    raise RuntimeError(
                        "prefix endpoint lacked a molecular incoming label"
                    )
                current = merged["isReplicator"].astype(float).to_numpy()
                next_label = np.concatenate((current[1:], [np.nan]))
                values = pd.to_numeric(merged["emergence"], errors="coerce").to_numpy(
                    float
                )
                for alignment, aligned in (
                    ("CURRENT_ENDPOINT", current),
                    ("NEXT_ENDPOINT", next_label),
                ):
                    result = association_summary(values, aligned)
                    rows.append(
                        {
                            "candidateId": candidate_id,
                            "matrixIndex": int(matrix_index),
                            "trajectoryId": str(merged["trajectoryId"].iloc[0]),
                            "implementationId": "PHIRL_REGULARIZED_SOURCE",
                            "metric": "emergence",
                            "temporalMode": "PAST_ONLY_PREFIX_ENDPOINT",
                            "labelId": label_id,
                            "alignment": alignment,
                            **result,
                        }
                    )
                    mask = np.isfinite(values) & np.isfinite(aligned)
                    payloads[(candidate_id, int(matrix_index), label_id, alignment)] = (
                        values[mask],
                        aligned[mask],
                    )
    return pd.DataFrame(rows), payloads


def summarize(
    details: pd.DataFrame, payloads: dict[tuple[Any, ...], Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    inference_rows = []
    for (candidate, label_id, alignment), group in details.groupby(
        ["candidateId", "labelId", "alignment"], sort=True
    ):
        rhos = group["rho"].dropna().to_numpy(float)
        differences = group["meanDifference"].dropna().to_numpy(float)
        pvalues = pd.to_numeric(group["ordinaryTwoSidedP"], errors="coerce").to_numpy(
            float
        )
        rho_all = pd.to_numeric(group["rho"], errors="coerce").to_numpy(float)
        summary = {
            "candidateId": candidate,
            "implementationId": "PHIRL_REGULARIZED_SOURCE",
            "metric": "emergence",
            "temporalMode": "PAST_ONLY_PREFIX_ENDPOINT",
            "labelId": label_id,
            "alignment": alignment,
            "trajectoryCount": len(group),
            "definedCorrelationCount": len(rhos),
            "positiveCorrelationCount": int(np.count_nonzero(rhos > 0)),
            "positiveCorrelationFraction": float(np.mean(rhos > 0)),
            "positiveSignificantCount": int(
                np.count_nonzero((rho_all > 0) & (pvalues < 0.05))
            ),
            "medianCorrelation": float(np.median(rhos)),
            "meanCorrelation": float(np.mean(rhos)),
            "definedDriftCount": len(differences),
            "higherDuringReplicationCount": int(np.count_nonzero(differences > 0)),
            "higherDuringReplicationFraction": float(np.mean(differences > 0)),
            "medianMeanDifference": float(np.median(differences)),
        }
        summaries.append(summary)
        boot_rng = np.random.default_rng(
            derive_seed("prefix_audit", candidate, label_id, alignment, "bootstrap")
        )
        shift_rng = np.random.default_rng(
            derive_seed("prefix_audit", candidate, label_id, alignment, "shift")
        )
        boot = np.median(
            rhos[
                boot_rng.integers(0, len(rhos), size=(BOOTSTRAP_REPLICATES, len(rhos)))
            ],
            axis=1,
        )
        shift_columns = []
        for matrix_index in sorted(group["matrixIndex"].astype(int)):
            values, labels = payloads[(candidate, matrix_index, label_id, alignment)]
            correlations, _ = _circular_vectors(values, labels)
            if len(correlations) > 1:
                offsets = shift_rng.integers(
                    1, len(correlations), size=SHIFT_REPLICATES
                )
                shift_columns.append(correlations[offsets])
        null = (
            np.median(np.column_stack(shift_columns), axis=1)
            if shift_columns
            else np.asarray([])
        )
        inference_rows.append(
            {
                **summary,
                "bootstrapLower95": float(np.quantile(boot, 0.025)),
                "bootstrapUpper95": float(np.quantile(boot, 0.975)),
                "circularShiftPositiveP": float(
                    (1 + np.count_nonzero(null >= summary["medianCorrelation"]))
                    / (1 + len(null))
                ),
                "bootstrapReplicates": BOOTSTRAP_REPLICATES,
                "circularShiftReplicates": SHIFT_REPLICATES,
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(inference_rows)


def append_ledger(summaries: pd.DataFrame) -> None:
    path = STEP_ROOT / "chronological_search_ledger.csv"
    existing = pd.read_csv(path)
    start = int(existing["attemptSequence"].max()) + 1
    rows = []
    for offset, ((label_id, alignment), group) in enumerate(
        summaries.groupby(["labelId", "alignment"], sort=True)
    ):
        rows.append(
            {
                "attemptSequence": start + offset,
                "attemptId": f"S13X-PREFIX-{label_id}-{alignment}",
                "phase": "POST_LEAD_PAST_ONLY_PREFIX_AUDIT",
                "choiceFamily": "TEMPORAL_FITTING_DEPENDENCE",
                "specification": json.dumps(
                    {
                        "implementationId": "PHIRL_REGULARIZED_SOURCE",
                        "metric": "emergence",
                        "temporalMode": "PAST_ONLY_PREFIX_ENDPOINT",
                        "labelId": label_id,
                        "alignment": alignment,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "evidenceTier": "ADAPTIVE_PAST_ONLY_SOURCE_RECONSTRUCTION",
                "outcome": "; ".join(
                    f"{row.candidateId}:medianRho={row.medianCorrelation:.6f},"
                    f"positiveFraction={row.positiveCorrelationFraction:.6f}"
                    for row in group.itertuples(index=False)
                ),
                "negativeResult": bool(
                    not (
                        (group["medianCorrelation"] > 0).all()
                        and (group["positiveCorrelationFraction"] > 0.5).all()
                    )
                ),
                "selectionUse": (
                    "Directly separates completed-fit resemblance from past-only support; "
                    "adaptive, not confirmatory."
                ),
            }
        )
    write_csv(path, pd.concat([existing, pd.DataFrame(rows)], ignore_index=True))


def main() -> None:
    prefix = pd.read_parquet(PREFIX_INPUT)
    prefix = prefix[
        (prefix["implementationId"] == "PHIRL_REGULARIZED_SOURCE")
        & (prefix["status"] == "ELIGIBLE")
        & (prefix["priorLockedClockTransitions"] >= 256)
    ].copy()
    details, payloads = build_details(prefix)
    summaries, inference = summarize(details, payloads)
    suffix = pd.read_parquet(SUFFIX_INPUT)
    executed = suffix[suffix["sentinel"] != "non_sentinel"]
    suffix_passed = bool(
        suffix["structuralExact"].astype(bool).all()
        and executed["resultExact"].eq(True).all()
        and len(executed) == 3552
    )
    available_tasks = {
        (str(row.candidateId), int(row.matrixIndex))
        for row in prefix[["candidateId", "matrixIndex"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    expected_tasks = {
        (candidate_id, matrix_index)
        for candidate_id in CANDIDATE_IDS
        for matrix_index in range(100)
    }
    unavailable_tasks = sorted(expected_tasks - available_tasks)
    write_parquet(STEP_ROOT / "prefix_audit_trajectory_results.parquet", details)
    write_csv(STEP_ROOT / "prefix_audit_results.csv", summaries)
    write_csv(STEP_ROOT / "prefix_audit_inference.csv", inference)
    append_ledger(summaries)
    validation = {
        "schema": "eidosoma.e01.s13x_prefix_audit_validation.v1",
        "researchStepId": "S13X",
        "adaptiveAfterRetrospectiveLead": True,
        "outcomePreviewedDuringInteractiveGapTriage": True,
        "prefixInput": str(PREFIX_INPUT),
        "suffixInput": str(SUFFIX_INPUT),
        "eligiblePrefixRowCount": len(prefix),
        "availableTrajectoryTaskCount": len(available_tasks),
        "unavailableTrajectoryTasks": [
            {
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "reason": "frozen_task_has_no_eligible_prefix_endpoint",
            }
            for candidate_id, matrix_index in unavailable_tasks
        ],
        "trajectoryResultCount": len(details),
        "summaryCount": len(summaries),
        "inferenceCount": len(inference),
        "incomingLabelsAtEligibleEndpointsUseOnlyCurrentAndPreviousStates": True,
        "initialStateFutureAdjacentConventionExcludedFromEveryEligibleEndpoint": True,
        "executedSuffixSentinelCount": len(executed),
        "reusedSuffixValidationPassed": suffix_passed,
        "passed": bool(
            len(details) == len(available_tasks) * len(LABEL_IDS) * len(ALIGNMENTS)
            and len(summaries) == len(CANDIDATE_IDS) * len(LABEL_IDS) * len(ALIGNMENTS)
            and len(inference) == len(summaries)
            and unavailable_tasks
            == [("S12F-CANDIDATE-02", 72), ("S12F-CANDIDATE-03", 72)]
            and suffix_passed
        ),
    }
    write_json(STEP_ROOT / "prefix_audit_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError("S13X prefix audit validation failed")
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
