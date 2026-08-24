#!/usr/bin/env python3
"""Execute only the prospectively locked E01/S19-L02 label-definition loop."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_clean_directional_confirmation.core import fixed_label_spec
from e01_creative_directional_search.core import label_trajectory
from e01_s19_replicator_definition.core import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_IDS,
    COMPARATOR_LABEL_ID,
    LABEL_DEFINITIONS,
    LOOP_ID,
    PAPER_TARGETS,
    VERSION,
    closer_dimension_count,
    derive_seed128,
    fingerprint_from_labels,
    paper_fingerprint_distance,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L02"
CACHE_ROOT = Path("/cache/e01_s19_l02")
TRAJECTORY_CACHE = CACHE_ROOT / "label_trajectory_outputs"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
PREREG = REPO_ROOT / "configs/e01/s19_l02_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l02_method_lock.json"
LABEL_BY_ID = {item.label_id: item for item in LABEL_DEFINITIONS}
LABEL_ORDER = {item.label_id: item.ordinal for item in LABEL_DEFINITIONS}
FINGERPRINT_METRICS = (
    "persistence",
    "occupancy",
    "consistency",
    "firstOnsetRawIndex0",
    "firstOnsetNormalized",
    "entryCount",
    "exitCount",
    "episodeCount",
    "meanEpisodeDuration",
    "medianEpisodeDuration",
    "longestEpisode",
    "postFissionReplicatorFraction",
    "postFissionEpisodeCount",
    "sameReferenceReentryCount",
    "sameReferenceTemporalSpanNormalized",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json(value: object) -> str:
    return json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(canonical_json(list(array.shape)).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def bool_codes(values: Iterable[Any]) -> np.ndarray:
    result = []
    for value in values:
        if value is None or pd.isna(value):
            result.append(-1)
        else:
            result.append(1 if bool(value) else 0)
    return np.asarray(result, dtype=np.int8)


def _normalize_frame(frame: pd.DataFrame, label_id: str) -> pd.DataFrame:
    definition = LABEL_BY_ID[label_id]
    result = frame.copy()
    if label_id == "PF_HISTORICAL_ADJACENT_AVERAGE_H090":
        initial = result["generation"].eq(0)
        result.loc[initial, "isReplicator"] = None
        result.loc[initial, "labelScore"] = np.nan
        result["labelStatus"] = np.where(initial, "INELIGIBLE_NO_GENERATION_LABEL", "ELIGIBLE")
        result["ineligibilityReason"] = np.where(
            initial, "initial_state_or_undefined_historical_label", None
        )
    else:
        result["labelStatus"] = "ELIGIBLE"
        result["ineligibilityReason"] = None
    result["researchStepId"] = LOOP_ID
    result["labelId"] = label_id
    result["labelFamily"] = definition.family_name
    result["labelEvidenceTier"] = definition.evidence_class
    result["temporalScope"] = definition.temporal_scope
    return result[
        [
            "researchStepId",
            "candidateId",
            "trajectoryId",
            "matrixIndex",
            "labelId",
            "labelFamily",
            "labelEvidenceTier",
            "temporalScope",
            "selectedSequenceIndex",
            "rawObservationIndex",
            "generation",
            "observationKind",
            "isReplicator",
            "labelScore",
            "labelStatus",
            "ineligibilityReason",
        ]
    ]


def _frame_identity(frame: pd.DataFrame) -> dict[str, str]:
    ordered = frame.sort_values("selectedSequenceIndex", kind="stable")
    status_bytes = "\x1f".join(ordered["labelStatus"].astype(str)).encode()
    kind_bytes = "\x1f".join(ordered["observationKind"].astype(str)).encode()
    return {
        "labelSha256": sha256_array(bool_codes(ordered["isReplicator"])),
        "scoreSha256": sha256_array(ordered["labelScore"].to_numpy(dtype=np.float64)),
        "sequenceSha256": sha256_array(ordered["selectedSequenceIndex"].to_numpy(dtype=np.int64)),
        "rawIndexSha256": sha256_array(ordered["rawObservationIndex"].to_numpy(dtype=np.int64)),
        "generationSha256": sha256_array(ordered["generation"].to_numpy(dtype=np.int64)),
        "statusSha256": hashlib.sha256(status_bytes).hexdigest(),
        "kindSha256": hashlib.sha256(kind_bytes).hexdigest(),
    }


def _trajectory_worker(record: dict[str, Any]) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    path = Path(record["cachePath"])
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for definition in LABEL_DEFINITIONS:
        spec = fixed_label_spec(definition.label_id)
        first_raw, first_diagnostic = label_trajectory(
            trajectory, spec, clock_id=str(record["clockId"])
        )
        second_raw, second_diagnostic = label_trajectory(
            trajectory, spec, clock_id=str(record["clockId"])
        )
        first = _normalize_frame(first_raw, definition.label_id)
        second = _normalize_frame(second_raw, definition.label_id)
        first_identity = _frame_identity(first)
        second_identity = _frame_identity(second)
        diagnostic_fields = ("referenceSize", "selectedK", "silhouette")
        diagnostics_equal = all(
            (
                first_diagnostic.get(key) == second_diagnostic.get(key)
                or (
                    first_diagnostic.get(key) is not None
                    and second_diagnostic.get(key) is not None
                    and np.isclose(
                        float(first_diagnostic[key]),
                        float(second_diagnostic[key]),
                        rtol=0.0,
                        atol=0.0,
                    )
                )
            )
            for key in diagnostic_fields
        )
        replay_pass = first_identity == second_identity and diagnostics_equal
        replay_rows.append(
            {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "labelId": definition.label_id,
                **{f"first{key[0].upper()}{key[1:]}": value for key, value in first_identity.items()},
                **{f"second{key[0].upper()}{key[1:]}": value for key, value in second_identity.items()},
                "diagnosticsEqual": diagnostics_equal,
                "exactReplayPassed": replay_pass,
            }
        )
        diagnostic_rows.append(
            {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "labelId": definition.label_id,
                "referenceSize": first_diagnostic.get("referenceSize"),
                "selectedK": first_diagnostic.get("selectedK"),
                "silhouette": first_diagnostic.get("silhouette"),
            }
        )
        frames.append(first)
    combined = pd.concat(frames, ignore_index=True)
    output = TRAJECTORY_CACHE / str(record["candidateId"]) / f"M{int(record['matrixIndex']):03d}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False, compression="zstd")
    return {
        "candidateId": record["candidateId"],
        "matrixIndex": int(record["matrixIndex"]),
        "trajectoryId": record["trajectoryId"],
        "cacheOutput": str(output),
        "diagnostics": diagnostic_rows,
        "replay": replay_rows,
        "wallSeconds": time.perf_counter() - started_wall,
        "cpuSeconds": time.process_time() - started_cpu,
        "success": all(row["exactReplayPassed"] for row in replay_rows),
    }


def validate_execution_lock() -> dict[str, Any]:
    repository = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    replay = json.loads((LOOP_ROOT / "preanalysis_replay_validation.json").read_text())
    benchmark = json.loads((LOOP_ROOT / "compute_benchmark.json").read_text())
    s18 = json.loads((LOOP_ROOT / "s18_immutable_validation.json").read_text())
    l01 = json.loads((LOOP_ROOT / "immutable_prior_validation.json").read_text())
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    clean = not bool(git("status", "--porcelain=v1"))
    config_hashes = {
        "preregistration": sha256_file(PREREG),
        "artifactPreregistration": sha256_file(LOOP_ROOT / "preregistration.yaml"),
        "methodLock": sha256_file(METHOD_LOCK),
        "artifactMethodLock": sha256_file(LOOP_ROOT / "method_lock.json"),
    }
    passed = bool(
        repository["passed"]
        and replay["passed"]
        and benchmark["gatePassed"]
        and s18["passed"]
        and l01["passed"]
        and head == remote == repository["head"]
        and clean
        and config_hashes["preregistration"] == config_hashes["artifactPreregistration"]
        and config_hashes["methodLock"] == config_hashes["artifactMethodLock"]
    )
    return {
        "schema": "eidosoma.e01.s19_l02_execution_lock_validation.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "preparedHead": repository["head"],
        "cleanWorktree": clean,
        "configHashes": config_hashes,
        "preanalysisReplayPassed": replay["passed"],
        "immutablePriorPassed": s18["passed"] and l01["passed"],
        "benchmarkPassed": benchmark["gatePassed"],
        "passed": passed,
    }


def execute_labels(
    manifest: pd.DataFrame, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = manifest.sort_values(["matrixIndex", "candidateId"], kind="stable").to_dict(orient="records")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_trajectory_worker, record): record for record in records}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (row["matrixIndex"], row["candidateId"]))
    execution = pd.DataFrame(
        [
            {
                "candidateId": row["candidateId"],
                "matrixIndex": row["matrixIndex"],
                "trajectoryId": row["trajectoryId"],
                "success": row["success"],
                "wallSeconds": row["wallSeconds"],
                "cpuSeconds": row["cpuSeconds"],
                "cacheOutput": row["cacheOutput"],
            }
            for row in results
        ]
    )
    replay = pd.DataFrame([item for row in results for item in row["replay"]])
    diagnostics = pd.DataFrame([item for row in results for item in row["diagnostics"]])
    frames = [pd.read_parquet(row["cacheOutput"]) for row in results]
    labels = pd.concat(frames, ignore_index=True)
    labels["isReplicator"] = pd.array(labels["isReplicator"], dtype="boolean")
    return (
        labels,
        diagnostics,
        replay.merge(
            execution[["candidateId", "matrixIndex", "success"]],
            on=["candidateId", "matrixIndex"],
            how="left",
        ),
        execution,
    )


def replace_historical_with_frozen(
    computed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain the exact frozen S13Y propagation and score fields.

    The frozen S13Y label stores historical incoming H as its diagnostic score,
    while the boolean state is source technique-1's local average-H decision.
    Fresh technique-1 booleans must agree; the frozen diagnostic serialization is
    then retained byte-for-value rather than silently changing its meaning.
    """

    label_id = "PF_HISTORICAL_ADJACENT_AVERAGE_H090"
    fresh = computed.loc[computed["labelId"].eq(label_id)].sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
    )
    frozen_all = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    frozen = frozen_all.loc[
        frozen_all["labelId"].eq("HISTORICAL_H090_REPLICATOR")
    ].sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
    )
    rows = []
    for candidate in CANDIDATE_IDS:
        for matrix in range(100):
            left = fresh.loc[
                fresh["candidateId"].eq(candidate)
                & fresh["matrixIndex"].eq(matrix)
            ]
            right = frozen.loc[
                frozen["candidateId"].eq(candidate)
                & frozen["matrixIndex"].eq(matrix)
            ]
            clock_equal = len(left) == len(right) and np.array_equal(
                left["selectedSequenceIndex"].to_numpy(dtype=np.int64),
                right["selectedSequenceIndex"].to_numpy(dtype=np.int64),
            )
            labels_equal = np.array_equal(
                bool_codes(left["isReplicator"]), bool_codes(right["isReplicator"])
            )
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix,
                    "labelId": label_id,
                    "freshTechnique1ClockPassed": clock_equal,
                    "freshTechnique1BooleanPassed": labels_equal,
                    "frozenIncomingHDiagnosticRetained": True,
                    "passed": bool(clock_equal and labels_equal),
                }
            )
    validation = pd.DataFrame(rows)
    if not validation["passed"].all():
        raise RuntimeError("fresh historical technique-1 labels differ from frozen S13Y")
    replacement = frozen.copy()
    definition = LABEL_BY_ID[label_id]
    replacement["researchStepId"] = LOOP_ID
    replacement["labelId"] = label_id
    replacement["labelFamily"] = definition.family_name
    replacement["labelEvidenceTier"] = definition.evidence_class
    replacement["temporalScope"] = definition.temporal_scope
    replacement = replacement[
        [
            "researchStepId",
            "candidateId",
            "trajectoryId",
            "matrixIndex",
            "labelId",
            "labelFamily",
            "labelEvidenceTier",
            "temporalScope",
            "selectedSequenceIndex",
            "rawObservationIndex",
            "generation",
            "observationKind",
            "isReplicator",
            "labelScore",
            "labelStatus",
            "ineligibilityReason",
        ]
    ]
    retained = computed.loc[~computed["labelId"].eq(label_id)]
    output = pd.concat([retained, replacement], ignore_index=True)
    output["isReplicator"] = pd.array(output["isReplicator"], dtype="boolean")
    return output, validation


def frozen_input_replay(labels: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    mapping = {
        COMPARATOR_LABEL_ID: COMPARATOR_LABEL_ID,
        "PF_HISTORICAL_ADJACENT_AVERAGE_H090": "HISTORICAL_H090_REPLICATOR",
    }
    rows = []
    for label_id, frozen_id in mapping.items():
        current = labels.loc[labels["labelId"].eq(label_id)].sort_values(
            ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
        )
        expected = frozen.loc[frozen["labelId"].eq(frozen_id)].sort_values(
            ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
        )
        for candidate in CANDIDATE_IDS:
            for matrix in range(100):
                left = current.loc[
                    current["candidateId"].eq(candidate) & current["matrixIndex"].eq(matrix)
                ]
                right = expected.loc[
                    expected["candidateId"].eq(candidate) & expected["matrixIndex"].eq(matrix)
                ]
                identity = len(left) == len(right)
                identity &= np.array_equal(
                    left["selectedSequenceIndex"].to_numpy(dtype=np.int64),
                    right["selectedSequenceIndex"].to_numpy(dtype=np.int64),
                )
                labels_equal = np.array_equal(bool_codes(left["isReplicator"]), bool_codes(right["isReplicator"]))
                scores_equal = np.array_equal(
                    left["labelScore"].to_numpy(dtype=np.float64),
                    right["labelScore"].to_numpy(dtype=np.float64),
                    equal_nan=True,
                )
                rows.append(
                    {
                        "candidateId": candidate,
                        "matrixIndex": matrix,
                        "labelId": label_id,
                        "frozenLabelId": frozen_id,
                        "rowIdentityPassed": identity,
                        "labelIdentityPassed": labels_equal,
                        "scoreIdentityPassed": scores_equal,
                        "passed": bool(identity and labels_equal and scores_equal),
                    }
                )
    return pd.DataFrame(rows)


def build_fingerprints(labels: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = labels.sort_values(
        ["candidateId", "matrixIndex", "labelId", "selectedSequenceIndex"], kind="stable"
    ).groupby(["candidateId", "matrixIndex", "trajectoryId", "labelId"], sort=False)
    for (candidate, matrix, trajectory, label_id), group in grouped:
        definition = LABEL_BY_ID[label_id]
        result = fingerprint_from_labels(
            sequence_indices=group["selectedSequenceIndex"].tolist(),
            labels=group["isReplicator"].astype(object).where(group["isReplicator"].notna(), None).tolist(),
            total_clock_count=len(group),
            observation_kinds=group["observationKind"].tolist(),
            global_reference=definition.global_reference,
        )
        diagnostic = diagnostics.loc[
            diagnostics["candidateId"].eq(candidate)
            & diagnostics["matrixIndex"].eq(int(matrix))
            & diagnostics["labelId"].eq(label_id)
        ].iloc[0]
        rows.append(
            {
                "candidateId": candidate,
                "matrixIndex": int(matrix),
                "trajectoryId": trajectory,
                "labelId": label_id,
                "labelOrdinal": definition.ordinal,
                "labelFamily": definition.family_name,
                "evidenceClass": definition.evidence_class,
                "temporalScope": definition.temporal_scope,
                "globalReference": definition.global_reference,
                "sourceGroundingGate": definition.source_grounding_gate,
                "unresolvedMaterialChoice": definition.unresolved_material_choice,
                "referenceSize": None if pd.isna(diagnostic["referenceSize"]) else int(diagnostic["referenceSize"]),
                "selectedK": None if pd.isna(diagnostic["selectedK"]) else int(diagnostic["selectedK"]),
                "silhouette": None if pd.isna(diagnostic["silhouette"]) else float(diagnostic["silhouette"]),
                **result,
            }
        )
    return pd.DataFrame(rows).sort_values(["labelOrdinal", "candidateId", "matrixIndex"], kind="stable")


def episode_table(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = labels.sort_values(
        ["candidateId", "matrixIndex", "labelId", "selectedSequenceIndex"], kind="stable"
    ).groupby(["candidateId", "matrixIndex", "trajectoryId", "labelId"], sort=False)
    for keys, group in grouped:
        candidate, matrix, trajectory, label_id = keys
        valid = group["isReplicator"].notna().to_numpy()
        indices = group["selectedSequenceIndex"].to_numpy(dtype=np.int64)[valid]
        values = group.loc[valid, "isReplicator"].to_numpy(dtype=bool)
        start = None
        prior = None
        episode_index = 0
        for index, value in zip(indices, values, strict=True):
            index = int(index)
            contiguous = prior is not None and index == prior + 1
            if value and (start is None or not contiguous):
                if start is not None and prior is not None:
                    rows.append({"candidateId": candidate, "matrixIndex": int(matrix), "trajectoryId": trajectory, "labelId": label_id, "episodeIndex": episode_index, "startIndex0": start, "endIndex0": prior, "duration": prior - start + 1})
                    episode_index += 1
                start = index
            elif not value and start is not None:
                assert prior is not None
                rows.append({"candidateId": candidate, "matrixIndex": int(matrix), "trajectoryId": trajectory, "labelId": label_id, "episodeIndex": episode_index, "startIndex0": start, "endIndex0": prior, "duration": prior - start + 1})
                episode_index += 1
                start = None
            prior = index
        if start is not None and prior is not None:
            rows.append({"candidateId": candidate, "matrixIndex": int(matrix), "trajectoryId": trajectory, "labelId": label_id, "episodeIndex": episode_index, "startIndex0": start, "endIndex0": prior, "duration": prior - start + 1})
    return pd.DataFrame(rows)


def aggregate_fingerprints(fingerprints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide_rows = []
    long_rows = []
    for (candidate, label_id), group in fingerprints.groupby(["candidateId", "labelId"], sort=False):
        row: dict[str, Any] = {
            "candidateId": candidate,
            "labelId": label_id,
            "labelOrdinal": LABEL_ORDER[label_id],
            "trajectoryCount": len(group),
            "definedConsistencyCount": int(group["consistency"].notna().sum()),
            "observedOnsetCount": int((~group["neverReplicator"]).sum()),
            "neverReplicatorCount": int(group["neverReplicator"].sum()),
            "nonreplicatingAtCutoffFraction": float(group["isNonreplicatingAtCutoff"].astype(float).mean()),
            "noReplicatorThroughCutoffFraction": float(group["noReplicatorObservedThroughCutoff"].astype(float).mean()),
        }
        for metric in FINGERPRINT_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            prefix = metric[0].upper() + metric[1:]
            row[f"defined{prefix}Count"] = len(finite)
            row[f"mean{prefix}"] = float(finite.mean()) if len(finite) else None
            row[f"median{prefix}"] = float(finite.median()) if len(finite) else None
            row[f"sd{prefix}"] = float(finite.std(ddof=1)) if len(finite) > 1 else None
            row[f"q025{prefix}"] = float(finite.quantile(0.025)) if len(finite) else None
            row[f"q975{prefix}"] = float(finite.quantile(0.975)) if len(finite) else None
            for statistic in ("mean", "median", "sd", "q025", "q975"):
                long_rows.append(
                    {
                        "candidateId": candidate,
                        "labelId": label_id,
                        "metric": metric,
                        "statistic": statistic,
                        "value": row.get(f"{statistic}{prefix}"),
                        "definedCount": len(finite),
                        "trajectoryCount": len(group),
                    }
                )
        scoring = {
            "persistence": row["meanPersistence"],
            "occupancy": row["meanOccupancy"],
            "consistency": row["meanConsistency"],
            "firstOnsetRawScore": float(group["firstOnsetRawScore"].mean()),
            "firstOnsetNormalizedScore": float(group["firstOnsetNormalizedScore"].mean()),
        }
        row["meanFirstOnsetRawScore"] = scoring["firstOnsetRawScore"]
        row["meanFirstOnsetNormalizedScore"] = scoring["firstOnsetNormalizedScore"]
        row["paperDistanceRaw"] = paper_fingerprint_distance(scoring, onset_mode="RAW")
        row["paperDistanceNormalized"] = paper_fingerprint_distance(scoring, onset_mode="NORMALIZED")
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows).sort_values(["labelOrdinal", "candidateId"], kind="stable")
    return wide, pd.DataFrame(long_rows)


def score_group(group: pd.DataFrame, onset_mode: str) -> float | None:
    summary = {
        "persistence": float(group["persistence"].mean()),
        "occupancy": float(group["occupancy"].mean()),
        "consistency": float(group["consistency"].dropna().mean()) if group["consistency"].notna().any() else None,
        "firstOnsetRawScore": float(group["firstOnsetRawScore"].mean()),
        "firstOnsetNormalizedScore": float(group["firstOnsetNormalizedScore"].mean()),
    }
    return paper_fingerprint_distance(summary, onset_mode=onset_mode)  # type: ignore[arg-type]


def paper_comparisons(
    fingerprints: pd.DataFrame, aggregate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    replicate_rows = []
    loo_rows = []
    for candidate in CANDIDATE_IDS:
        comparator = fingerprints.loc[
            fingerprints["candidateId"].eq(candidate) & fingerprints["labelId"].eq(COMPARATOR_LABEL_ID)
        ].sort_values("matrixIndex", kind="stable")
        comparator_agg = aggregate.loc[
            aggregate["candidateId"].eq(candidate) & aggregate["labelId"].eq(COMPARATOR_LABEL_ID)
        ].iloc[0]
        for definition in LABEL_DEFINITIONS[1:]:
            candidate_frame = fingerprints.loc[
                fingerprints["candidateId"].eq(candidate) & fingerprints["labelId"].eq(definition.label_id)
            ].sort_values("matrixIndex", kind="stable")
            if not np.array_equal(candidate_frame["matrixIndex"], comparator["matrixIndex"]):
                raise RuntimeError("matrix pairing changed")
            candidate_agg = aggregate.loc[
                aggregate["candidateId"].eq(candidate) & aggregate["labelId"].eq(definition.label_id)
            ].iloc[0]
            candidate_summary = {
                "persistence": candidate_agg["meanPersistence"],
                "occupancy": candidate_agg["meanOccupancy"],
                "consistency": candidate_agg["meanConsistency"],
                "firstOnsetRawScore": candidate_agg["meanFirstOnsetRawScore"],
                "firstOnsetNormalizedScore": candidate_agg["meanFirstOnsetNormalizedScore"],
            }
            comparator_summary = {
                "persistence": comparator_agg["meanPersistence"],
                "occupancy": comparator_agg["meanOccupancy"],
                "consistency": comparator_agg["meanConsistency"],
                "firstOnsetRawScore": comparator_agg["meanFirstOnsetRawScore"],
                "firstOnsetNormalizedScore": comparator_agg["meanFirstOnsetNormalizedScore"],
            }
            for mode in ("RAW", "NORMALIZED"):
                point = score_group(candidate_frame, mode)
                comparator_point = score_group(comparator, mode)
                if point is None or comparator_point is None:
                    improvement = None
                else:
                    improvement = float((comparator_point - point) / comparator_point) if comparator_point else None
                closer, structure = closer_dimension_count(candidate_summary, comparator_summary, onset_mode=mode)  # type: ignore[arg-type]
                rng = np.random.Generator(
                    np.random.PCG64DXSM(derive_seed128(candidate, definition.label_id, mode, "paired-paper-distance-bootstrap"))
                )
                diffs = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
                for replicate in range(BOOTSTRAP_REPLICATES):
                    sampled = rng.integers(0, len(candidate_frame), size=len(candidate_frame))
                    left = score_group(candidate_frame.iloc[sampled], mode)
                    right = score_group(comparator.iloc[sampled], mode)
                    diffs[replicate] = np.nan if left is None or right is None else left - right
                    replicate_rows.append(
                        {
                            "candidateId": candidate,
                            "labelId": definition.label_id,
                            "onsetMode": mode,
                            "replicate": replicate,
                            "paperDistanceDifferenceVsComparator": diffs[replicate],
                        }
                    )
                finite = diffs[np.isfinite(diffs)]
                ci = np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
                loo = []
                for omit in range(len(candidate_frame)):
                    keep = np.arange(len(candidate_frame)) != omit
                    left = score_group(candidate_frame.iloc[keep], mode)
                    right = score_group(comparator.iloc[keep], mode)
                    diff = np.nan if left is None or right is None else left - right
                    loo.append(diff)
                    loo_rows.append(
                        {
                            "candidateId": candidate,
                            "labelId": definition.label_id,
                            "onsetMode": mode,
                            "omittedMatrixIndex": int(candidate_frame.iloc[omit]["matrixIndex"]),
                            "paperDistanceDifferenceVsComparator": diff,
                        }
                    )
                comparison_rows.append(
                    {
                        "candidateId": candidate,
                        "labelId": definition.label_id,
                        "onsetMode": mode,
                        "paperDistance": point,
                        "comparatorDistance": comparator_point,
                        "distanceImprovementFraction": improvement,
                        "closerDimensionCount": closer,
                        "onsetOrConsistencyImproved": structure,
                        "bootstrapDifferenceCi95Low": float(ci[0]) if np.isfinite(ci[0]) else None,
                        "bootstrapDifferenceCi95High": float(ci[1]) if np.isfinite(ci[1]) else None,
                        "bootstrapImprovementProbability": float(np.mean(finite < 0)) if len(finite) else None,
                        "leaveOneOutDifferenceMin": float(np.nanmin(loo)),
                        "leaveOneOutDifferenceMax": float(np.nanmax(loo)),
                        "leaveOneOutAllImproved": bool(np.all(np.asarray(loo) < 0)),
                    }
                )
    return pd.DataFrame(comparison_rows), pd.DataFrame(replicate_rows), pd.DataFrame(loo_rows)


def label_overlap(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for candidate in CANDIDATE_IDS:
        for matrix in range(100):
            comparator = labels.loc[
                labels["candidateId"].eq(candidate)
                & labels["matrixIndex"].eq(matrix)
                & labels["labelId"].eq(COMPARATOR_LABEL_ID)
            ].sort_values("selectedSequenceIndex", kind="stable")
            for definition in LABEL_DEFINITIONS:
                current = labels.loc[
                    labels["candidateId"].eq(candidate)
                    & labels["matrixIndex"].eq(matrix)
                    & labels["labelId"].eq(definition.label_id)
                ].sort_values("selectedSequenceIndex", kind="stable")
                if not np.array_equal(current["selectedSequenceIndex"], comparator["selectedSequenceIndex"]):
                    raise RuntimeError("label-overlap clocks do not align")
                left_codes = bool_codes(current["isReplicator"])
                right_codes = bool_codes(comparator["isReplicator"])
                keep = (left_codes >= 0) & (right_codes >= 0)
                left = left_codes[keep] == 1
                right = right_codes[keep] == 1
                union = np.count_nonzero(left | right)
                intersection = np.count_nonzero(left & right)
                rows.append(
                    {
                        "candidateId": candidate,
                        "matrixIndex": matrix,
                        "labelId": definition.label_id,
                        "commonEligibleCount": int(np.count_nonzero(keep)),
                        "accuracyVsAdjacentH": float(np.mean(left == right)),
                        "jaccardVsAdjacentH": float(intersection / union) if union else 1.0,
                        "mismatchFractionVsAdjacentH": float(np.mean(left != right)),
                    }
                )
    trajectory = pd.DataFrame(rows)
    summary = (
        trajectory.groupby(["candidateId", "labelId"], as_index=False)
        .agg(
            medianAccuracyVsAdjacentH=("accuracyVsAdjacentH", "median"),
            meanAccuracyVsAdjacentH=("accuracyVsAdjacentH", "mean"),
            medianJaccardVsAdjacentH=("jaccardVsAdjacentH", "median"),
            meanJaccardVsAdjacentH=("jaccardVsAdjacentH", "mean"),
            medianMismatchFractionVsAdjacentH=("mismatchFractionVsAdjacentH", "median"),
        )
    )
    return trajectory, summary


def cross_candidate_agreement(fingerprints: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "persistence",
        "occupancy",
        "consistency",
        "firstOnsetRawScore",
        "firstOnsetNormalizedScore",
        "episodeCount",
        "meanEpisodeDuration",
        "longestEpisode",
        "isNonreplicatingAtCutoff",
        "noReplicatorObservedThroughCutoff",
    )
    rows = []
    for definition in LABEL_DEFINITIONS:
        left = fingerprints.loc[fingerprints["candidateId"].eq(CANDIDATE_IDS[0]) & fingerprints["labelId"].eq(definition.label_id)].set_index("matrixIndex")
        right = fingerprints.loc[fingerprints["candidateId"].eq(CANDIDATE_IDS[1]) & fingerprints["labelId"].eq(definition.label_id)].set_index("matrixIndex")
        for metric in metrics:
            x = pd.to_numeric(left[metric], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(right[metric], errors="coerce").to_numpy(dtype=np.float64)
            keep = np.isfinite(x) & np.isfinite(y)
            xv, yv = x[keep], y[keep]
            pearson = spearman = None
            if len(xv) >= 4 and np.ptp(xv) > 0 and np.ptp(yv) > 0:
                pearson = float(stats.pearsonr(xv, yv).statistic)
                spearman = float(stats.spearmanr(xv, yv).statistic)
            rows.append(
                {
                    "labelId": definition.label_id,
                    "metric": metric,
                    "pairedMatrixCount": len(xv),
                    "pearson": pearson,
                    "spearman": spearman,
                    "medianAbsoluteDifference": float(np.median(np.abs(xv - yv))) if len(xv) else None,
                    "meanCandidate2": float(np.mean(xv)) if len(xv) else None,
                    "meanCandidate3": float(np.mean(yv)) if len(yv) else None,
                }
            )
    return pd.DataFrame(rows)


def classify(
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    replay: pd.DataFrame,
    frozen_replay: pd.DataFrame,
    overlap_summary: pd.DataFrame,
) -> dict[str, Any]:
    classifications = []
    promoted = []
    for definition in LABEL_DEFINITIONS:
        candidate_rows = aggregate.loc[aggregate["labelId"].eq(definition.label_id)]
        exact_replay = bool(replay.loc[replay["labelId"].eq(definition.label_id), "exactReplayPassed"].all())
        frozen_subset = frozen_replay.loc[frozen_replay["labelId"].eq(definition.label_id)]
        if len(frozen_subset):
            exact_replay &= bool(frozen_subset["passed"].all())
        classes: list[str] = []
        gates: dict[str, bool] = {
            "notComparator": not definition.comparator_only,
            "sourceGrounding": definition.source_grounding_gate,
            "identityChangingAuthorChoiceResolved": definition.source_grounding_gate,
            "exactReplay": exact_replay,
            "definedConsistencyBothCandidates": bool((candidate_rows["definedConsistencyCount"] >= 95).all()),
            "observedOnsetBothCandidates": bool((candidate_rows["observedOnsetCount"] >= 95).all()),
        }
        if definition.comparator_only:
            classes.extend(["POSSIBLE_STABILITY_PROXY", "NOT_PROMOTABLE"])
            gates.update(
                {
                    "distanceImprovementBothModesBothCandidates": False,
                    "bootstrapBothModesBothCandidates": False,
                    "threeDimensionsIncludingStructureBothModesBothCandidates": False,
                    "leaveOneOutBothModesBothCandidates": False,
                }
            )
        else:
            subset = comparisons.loc[comparisons["labelId"].eq(definition.label_id)]
            distance_gate = bool((subset["distanceImprovementFraction"] >= 0.10).all()) and len(subset) == 4
            bootstrap_gate = bool((subset["bootstrapDifferenceCi95High"] < 0).all()) and len(subset) == 4
            dimension_gate = bool(
                (subset["closerDimensionCount"] >= 3).all()
                and subset["onsetOrConsistencyImproved"].all()
            ) and len(subset) == 4
            loo_gate = bool(subset["leaveOneOutAllImproved"].all()) and len(subset) == 4
            gates.update(
                {
                    "distanceImprovementBothModesBothCandidates": distance_gate,
                    "bootstrapBothModesBothCandidates": bootstrap_gate,
                    "threeDimensionsIncludingStructureBothModesBothCandidates": dimension_gate,
                    "leaveOneOutBothModesBothCandidates": loo_gate,
                }
            )
            directional = distance_gate and dimension_gate
            paper_match = directional and bool(
                (candidate_rows["paperDistanceRaw"] <= 1.0).all()
                and (candidate_rows["paperDistanceNormalized"] <= 1.0).all()
            )
            if paper_match:
                classes.append("EXPLORATORY_PAPER_MATCH")
            elif directional:
                classes.append("EXPLORATORY_DIRECTIONAL_MATCH")
            else:
                classes.append("EXPLORATORY_NON_SUPPORT")
            if definition.temporal_scope == "RETROSPECTIVE_COMPLETED_TRAJECTORY":
                classes.extend(["RETROSPECTIVE_ONLY_LEAD", "METHOD_DEPENDENT_LEAD"])
            if definition.unresolved_material_choice:
                classes.append("AUTHOR_AMBIGUITY_UNRESOLVED")
            overlap = overlap_summary.loc[overlap_summary["labelId"].eq(definition.label_id)]
            stability_proxy = bool(
                len(overlap) == 2
                and (
                    (overlap["medianAccuracyVsAdjacentH"] >= 0.95)
                    | (overlap["medianJaccardVsAdjacentH"] >= 0.95)
                ).all()
            )
            if stability_proxy:
                classes.append("POSSIBLE_STABILITY_PROXY")
            promotable = all(gates.values())
            if promotable and len(promoted) < 2:
                classes.append("PROMOTABLE_TO_S20")
                promoted.append(definition.label_id)
            else:
                classes.append("NOT_PROMOTABLE")
        classifications.append(
            {
                "labelId": definition.label_id,
                "familyName": definition.family_name,
                "temporalScope": definition.temporal_scope,
                "classifications": list(dict.fromkeys(classes)),
                "promotionGates": gates,
                "promoted": definition.label_id in promoted,
                "allowedS20ClaimScope": (
                    "RETROSPECTIVE_PAPER_FACING_ONLY"
                    if definition.temporal_scope != "INCOMING_LOCAL_EXCEPT_INITIAL_DUPLICATES_FIRST_INCOMING_VALUE"
                    else "NONE_COMPARATOR"
                ),
            }
        )
    if len(promoted) > 2:
        raise RuntimeError("promotion limit exceeded")
    return {
        "schema": "eidosoma.e01.s19_l02_classification.v1",
        "researchStepId": LOOP_ID,
        "outcomeAccessed": True,
        "confirmatoryVerdictIssued": False,
        "labelClassifications": classifications,
        "promotedLeadCount": len(promoted),
        "promotedLeadIds": promoted,
        "defaultNextRecommendation": (
            "HUMAN_REVIEW_CONSIDER_ACTIVATE_S20_CONFIRMATION_OR_CLOSEOUT"
            if promoted
            else "ACTIVATE_S20_CLOSEOUT_ONLY_TWO_CONSECUTIVE_LOOPS_WITH_NO_PROMOTABLE_LEAD"
        ),
        "s20Activated": False,
        "laterLoopActivated": False,
    }


def format_number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "undefined"
    return f"{float(value):.{digits}f}"


def report_tables(aggregate: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[str, str]:
    selected = aggregate[
        [
            "candidateId",
            "labelId",
            "meanPersistence",
            "meanOccupancy",
            "meanConsistency",
            "meanFirstOnsetRawIndex0",
            "meanFirstOnsetNormalized",
            "meanEpisodeCount",
            "meanLongestEpisode",
            "nonreplicatingAtCutoffFraction",
            "noReplicatorThroughCutoffFraction",
            "paperDistanceRaw",
            "paperDistanceNormalized",
        ]
    ].copy()
    selected["candidateId"] = selected["candidateId"].str.replace("S12F-CANDIDATE-", "C", regex=False)
    selected["labelId"] = selected["labelId"].map(
        {
            "MOL_ADJACENT_INCOMING_H900": "Adjacent H>.9",
            "PF_DOMINANT_COMPONENT_CENTROID_H900": "Dominant centroid",
            "PF_EUCLIDEAN_KMEANS_DOMINANT": "Euclidean cluster",
            "PF_HISTORICAL_ADJACENT_AVERAGE_H090": "Historical T1",
        }
    )
    for column in selected.columns[2:]:
        selected[column] = selected[column].map(lambda value: format_number(value, 4))
    selected.columns = [
        "Cand.",
        "Label",
        "Persistence",
        "Occupancy",
        "Consistency",
        "Onset raw",
        "Onset norm.",
        "Episodes",
        "Longest",
        "Nonrep. at 25%",
        "No onset by 25%",
        "Distance raw",
        "Distance norm.",
    ]
    comp = comparisons.copy()
    comp["candidateId"] = comp["candidateId"].str.replace("S12F-CANDIDATE-", "C", regex=False)
    comp["labelId"] = comp["labelId"].map(
        {
            "PF_DOMINANT_COMPONENT_CENTROID_H900": "Dominant centroid",
            "PF_EUCLIDEAN_KMEANS_DOMINANT": "Euclidean cluster",
            "PF_HISTORICAL_ADJACENT_AVERAGE_H090": "Historical T1",
        }
    )
    comp = comp[
        [
            "candidateId",
            "labelId",
            "onsetMode",
            "distanceImprovementFraction",
            "closerDimensionCount",
            "onsetOrConsistencyImproved",
            "bootstrapDifferenceCi95Low",
            "bootstrapDifferenceCi95High",
            "leaveOneOutAllImproved",
        ]
    ]
    for column in ("distanceImprovementFraction", "bootstrapDifferenceCi95Low", "bootstrapDifferenceCi95High"):
        comp[column] = comp[column].map(lambda value: format_number(value, 4))
    comp.columns = ["Cand.", "Label", "Onset mode", "Distance gain", "Closer dims", "Structure improved", "Bootstrap low", "Bootstrap high", "All LOO improved"]
    return selected.to_markdown(index=False), comp.to_markdown(index=False)


def build_reports(
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    classification: dict[str, Any],
    validation: dict[str, Any],
    artifacts: list[str],
    runtime: dict[str, Any],
) -> tuple[str, str]:
    aggregate_table, comparison_table = report_tables(aggregate, comparisons)
    promoted = classification["promotedLeadIds"]
    outcome = "EXPLORATORY_CONSTRAINING_NO_PROMOTABLE_LEAD" if not promoted else "EXPLORATORY_PROMOTABLE_RETROSPECTIVE_LEAD"
    next_action = classification["defaultNextRecommendation"]
    caveat = (
        "No label passed the complete source-grounding and joint-fingerprint promotion gate; the two-loop default is S20 closeout-only."
        if not promoted
        else "Any promoted label remains exploratory and retrospective; only untouched S20 paper-facing confirmation is eligible."
    )
    top = f"""# S19-L02 — Replicator-definition temporal-fingerprint reconstruction

## Concise top summary

- **Research step ID:** S19-L02
- **Completion status:** COMPLETE; mandatory human-review boundary reached; S20 and every later loop remain inactive
- **Artifacts written:** {len(artifacts)} compact L02 evidence/report files plus append-only S19 root-ledger updates
- **Validation result:** {validation['validationResult']}
- **Outcome classification:** {outcome}
- **Caveats or blockers:** {caveat} The paper does not uniquely specify clustering, recurrence, threshold, reference, or molecular/generation alignment; completed-run clustering is future-dependent.
- **Recommended next action:** Human review must choose the next program action. Current default: `{next_action}`. Do not begin it automatically.
- **Lay summary:** This loop tested whether the gap between roughly 98% replication in the current adjacent-similarity label and the paper's roughly 88% state could be explained by a genuinely different definition of a replicator. It compared four fixed definitions and judged the whole temporal pattern—not occupancy alone. The analysis did not tune a threshold, generate simulations, or use causal-emergence results to pick a label.
"""
    report = top + f"""

## Frozen question and scientific boundary

The loop asked whether a recurring-attractor or historical GARD definition jointly improves the paper-facing fingerprint—persistence 716, occupancy 0.88, consistency 0.38, and first onset 37—relative to adjacent molecular `H>0.9`, in both simulator candidates. Because Table 1 prints onset as a percentage while its note says molecular steps, raw-step and normalized-onset distances were locked as separate analyses. Neither could replace the other.

This is an exploratory reconstruction on previously studied S13Y matrices. It does not revise S18, adjudicate Figures 3–6 under a new label, establish early warning, or establish causal control. No S19-L01 scientific value was repaired, rerun, reinterpreted, or used.

## Inputs

- Frozen S13Y trajectory manifest: 100 shared matrix identities, 200 candidate-specific trajectories, 100 completed fissions each.
- Candidate 2 and candidate 3 were analyzed separately. Pooling was not used for a primary gate.
- Frozen S13Y adjacent-H and historical technique-1 label arrays supplied exact replay comparators.
- Original v1 paper, S08 label contracts, S13X label implementations, S18 Matrix A, and pinned historical GARD v10 source identity supplied context and provenance.
- New GARD trajectories: **0**. New PhiRL/emergence values: **0**. GPU use: **0**.

## Detailed methods

### Pre-outcome lock and replay gate

The complete layout, four labels, compositional coordinates, distance/linkage, missing-data rules, target-distance formula, seeds, bootstrap, outlier checks, and promotion gates were committed and pushed before new label outcomes were opened. The preanalysis gate then reloaded all 200 trajectory caches and required exact candidate/trajectory identities, selected molecular clocks, adjacent-H float arrays, and strict `H>0.9` labels. Any mismatch would have failed the loop closed.

### Exactly four label families

1. **Adjacent molecular `H>0.9` comparator.** Incoming consecutive-state cosine H on the selected molecular clock. This is ordinary local smoothness, not a global attractor definition.
2. **Dominant recurring-composition centroid.** The frozen S13X completed-run post-fission cosine-component/centroid implementation. It is retrospective and method-dependent; the paper says Euclidean and omits the exact graph and centroid mechanics.
3. **Recurring Euclidean composition cluster.** The frozen S13X completed-run Euclidean silhouette/K-means implementation. It is retrospective and method-dependent; the paper omits K-means, K selection, and tie rules.
4. **Historical technique-1 compotype/non-drift.** The frozen historical GARD v10 adjacent post-fission average-H rule, propagated onto the molecular clock. The initial state remained explicitly ineligible. Interior labels depend on the outgoing neighbor, so this is not an online current-state label.

No `H>0.97` candidate or threshold grid was computed. The earlier 0.97 sensitivity served only as motivation already known before lock; it was not an L02 result or selection option.

### Temporal fingerprint

For every matrix and label, the loop retained occupancy, persistence, raw and normalized first onset, consecutive-label Pearson consistency, entries/exits, episode count and durations, longest episode, state and no-onset status at the 25% cutoff, and post-fission recurrence diagnostics. Full-run labels were explicitly marked retrospective.

The paper-distance score was the root mean square of errors standardized by the paper's control-table plus/minus values. It used persistence, occupancy, consistency, and one onset interpretation. Runs without onset retained null observed onset but received a right-censored score of total clock length (raw) or 1.0 (normalized), preventing missing onsets from improving the score. Undefined consistency was never imputed and failed promotion if fewer than 95/100 runs were defined.

Promotion required, in **both candidates and both onset interpretations**, at least 10% lower distance than adjacent H, a paired matrix-bootstrap 95% upper bound below zero, at least three of four target dimensions closer including consistency or onset, improvement under every leave-one-matrix-out omission, exact replay, adequate defined trajectories, and source-grounded label identity. Occupancy alone could not pass.

## Results

### Candidate-specific full temporal fingerprints

{aggregate_table}

### Joint fingerprint improvement relative to adjacent H

{comparison_table}

The complete distributional summaries, raw/normalized distance bootstraps, leave-one-out checks, cutoff measures, episode data, label overlap with exact H, recurrence diagnostics, and cross-candidate comparisons are machine-readable. Directional resemblance was evaluated even where exact numerical agreement failed, but a favorable direction in one candidate could not rescue the other.

### Classification and promotion

Promoted lead count: **{len(promoted)}**. Promoted IDs: **{', '.join(promoted) if promoted else 'none'}**.

"""
    for item in classification["labelClassifications"]:
        report += f"- `{item['labelId']}`: {', '.join(item['classifications'])}; promoted={str(item['promoted']).lower()}.\n"
    report += f"""

## Robustness and falsification

- Exact independent re-execution of every label on every trajectory: {validation['allLabelReplay']}.
- Frozen adjacent-H and historical-label identity: {validation['frozenInputReplay']}.
- Candidate 2 and candidate 3 remained separate through all primary calculations: PASS.
- Paired bootstrap unit: catalytic matrix; molecular rows were not treated as replicates.
- Leave-one-matrix-out influence: retained for every alternative label and both onset interpretations.
- Adjacent-H overlap: retained to identify definitions that remain proxies for ordinary local stability.
- New trajectories, emergence, association, prediction, and intervention outcomes used for selection: none.
- Immutable S01–S18/V1/V2/S19-L01 validation: {validation['immutablePrior']}.

## Commands and runtime

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 pytest -q tests/e01/test_s19_l02.py
git commit <pre-outcome L02 lock> && git push origin eidosoma/groups/42
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/prepare_s19_l02_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l02.py --workers 8
```

Scientific CPU time was {runtime['scientificCpuHours']:.6f} hours and wall time was {runtime['wallHours']:.6f} hours. Retained and temporary storage remained within the locked ceilings. CPU float64 was authoritative.

## Validation

Overall validation: **{validation['validationResult']}**. The report was rendered twice from machine-readable results and matched exactly. Required schemas, row counts, label cardinality, candidate cardinality, hashes, storage ceilings, replay, and promotion-limit checks passed.

## Caveats, blockers, and limitations

1. The paper's phrase “most recurring composition” does not identify a unique centroid, medoid, cluster algorithm, threshold, persistence rule, or tie rule. The two cluster implementations are fixed forensic reconstructions, not recovered author code.
2. Completed-run centroid and K-means labels use future observations and are eligible only for retrospective paper-facing interpretation.
3. Historical technique 1 is source-traceable to a public older GARD implementation, not the unavailable target-paper implementation; its outgoing-neighbor term is not cutoff-causal.
4. The paper's onset unit is internally inconsistent. Raw and normalized results remain separate.
5. The target-distance calculation necessarily uses known paper fingerprints and therefore can overfit those fingerprints. Untouched S20 data would be required for any confirmation.
6. No label was evaluated by its association with emergence or by downstream prediction/intervention performance, by design.
7. S18 and S19-L01 classifications remain unchanged. L02 is an additive exploratory record.

## Provenance

- Repository commit fixed before outcomes: `{runtime['repositoryCommit']}` on `eidosoma/groups/42`, equal to the pushed remote at access.
- Historical GARD v10: commit `86dff6320d5ae91b4e831471079ff46749b14df9`; no detected repository license; identity/hash only.
- Original paper: arXiv `2607.28250v1`, retained SHA-256 in `input_manifest.json`.
- S13Y trajectory and label hashes: `input_manifest.json` and `preanalysis_replay_evidence.parquet`.
- Complete source relationships and licensing boundaries: root `source_search_ledger.parquet` and L02 `source_snapshot_manifest.json`.

## Recommended next action and mandatory boundary

Return control for human review now. Recommended choice: `{next_action}`. This recommendation is not activation. Do not start S19-L03, S20, E02, author contact, or report-bundle generation automatically.
"""
    decision = top + f"""

## Decision evidence

Promoted leads: **{', '.join(promoted) if promoted else 'none'}**.

{comparison_table}

## Human choice required

The loop is frozen. The human reviewer must issue a new decision before any further scientific work. The current scientific recommendation is `{next_action}`. S20 remains defined but inactive.
"""
    return report, decision


def append_postloop_ledger(classification: dict[str, Any], aggregate: pd.DataFrame, timestamp: str) -> None:
    path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    existing = pd.read_parquet(path)
    if existing["loopId"].eq(LOOP_ID).sum() != 1:
        raise RuntimeError("expected exactly one L02 pre-loop ledger row")
    promoted = classification["promotedLeadIds"]
    compact = aggregate[
        ["candidateId", "labelId", "meanOccupancy", "meanConsistency", "meanFirstOnsetRawIndex0"]
    ].to_dict(orient="records")
    row = {
        "ledgerSequence": int(existing["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "POST_LOOP_LEARNING_AND_HUMAN_REVIEW_BOUNDARY",
        "beliefBeforeLoop": "A structurally recurring-attractor label, rather than a stricter adjacent-H threshold, might recover the paper's joint temporal fingerprint.",
        "motivatingEvidence": "Adjacent H saturated occupancy and immediate onset; paper wording and historical source imply recurring/global or generation-level structure.",
        "failureOrAmbiguityTargeted": "The 88% versus 98% occupancy discrepancy together with onset, consistency, episodes, and cutoff eligibility.",
        "selectedHypotheses": "Exactly four locked definitions; no threshold grid or downstream outcome selection.",
        "learned": canonical_json({"promotedLeadIds": promoted, "aggregateFingerprint": compact}),
        "weakenedHypotheses": "Any family failing joint raw/normalized cross-candidate fingerprint or source-grounding gates; occupancy-only matching remains rejected.",
        "remainingPlausibleHypotheses": "Only promoted labels, if any, remain eligible for untouched retrospective S20 confirmation; unavailable author mechanics remain unresolved.",
        "proposedNextTest": classification["defaultNextRecommendation"],
        "informationGainRationale": "The loop separated threshold prevalence from temporal structure and tested the upstream label without using emergence or L01 outcomes.",
        "appendOnly": True,
    }
    pd.concat([existing, pd.DataFrame([row])[existing.columns]], ignore_index=True).to_parquet(path, index=False)
    md = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    with md.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Entry 004 — S19-L02 post-loop learning and human-review boundary\n\n"
            f"- **What was learned:** Promoted leads: {', '.join(promoted) if promoted else 'none'}. The complete joint fingerprints are in `loops/L02/fingerprint_results.parquet`; occupancy alone was never sufficient.\n"
            "- **Hypotheses weakened:** Every definition that failed the cross-candidate raw/normalized temporal-fingerprint or source-grounding gates.\n"
            "- **Hypotheses remaining plausible:** Only explicitly promoted retrospective leads, if any; exact author label identity remains unavailable without source.\n"
            f"- **Proposed next action:** `{classification['defaultNextRecommendation']}`.\n"
            "- **Why another loop would or would not add information:** Two consecutive loops now have no promotable lead if the promoted list is empty; the standing default is terminal S20 closeout rather than adding more opportunities for a favorable result. Any override requires a new written scientific rationale.\n"
        )


def update_root_state(classification: dict[str, Any], full_report: str, artifacts: list[str]) -> None:
    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(loop_path.read_text(encoding="utf-8"))
    matches = [item for item in registry["loops"] if item["loopId"] == LOOP_ID]
    if len(matches) != 1:
        raise RuntimeError("L02 loop registry identity changed")
    matches[0].update(
        {
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "outcomeAccessed": True,
            "completed": True,
            "eligibleScientificResults": True,
            "promotedLeadCount": classification["promotedLeadCount"],
        }
    )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    loop_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["pendingDecision"] = "POST_S19_L02_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(full_report, encoding="utf-8")
    write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "researchStepId": LOOP_ID,
            "stepNumber": 19,
            "success": True,
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "artifactsWritten": artifacts,
            "validationResult": "PASS_ALL_LOCK_REPLAY_IMMUTABILITY_STORAGE_AND_REGENERATION_CHECKS",
            "caveatsOrBlockers": [
                "exploratory_previously_studied_matrices",
                "paper_does_not_uniquely_define_cluster_or_reference_mechanics",
                "completed_run_cluster_labels_are_retrospective",
                "S20_remains_inactive",
            ],
            "recommendedNextAction": classification["defaultNextRecommendation"],
        },
    )


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def artifact_manifest(root: Path, required: list[str], schema: str) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    observed = {row["path"] for row in files}
    missing = sorted(set(required) - observed)
    return {
        "schema": schema,
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(row["bytes"] for row in files),
        "files": files,
        "requiredFiles": required,
        "missing": missing,
        "passed": not missing,
    }


def root_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(item for item in ARTIFACT_ROOT.rglob("*") if item.is_file() and item != ARTIFACT_ROOT / "artifact_manifest.json"):
        files.append(
            {"path": str(path.relative_to(ARTIFACT_ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    required = [
        "continuation_decision.md",
        "s18_immutable_baseline.json",
        "self_improvement_ledger.parquet",
        "SELF_IMPROVEMENT_LEDGER.md",
        "candidate_registry.parquet",
        "source_search_ledger.parquet",
        "source_search_report.md",
        "loop_registry.yaml",
        "human_review_history.json",
        "s19_status.json",
        "research_step_full_results.md",
        "loops/L02/S19_L02_FULL_RESULTS.md",
        "loops/L02/classification.json",
        "loops/L02/artifact_manifest.json",
    ]
    observed = {row["path"] for row in files}
    return {
        "schema": "eidosoma.e01.s19_artifact_manifest.v2",
        "root": str(ARTIFACT_ROOT),
        "fileCount": len(files),
        "totalBytes": sum(row["bytes"] for row in files),
        "files": files,
        "requiredFiles": required,
        "missing": sorted(set(required) - observed),
        "passed": set(required).issubset(observed),
    }


def main(workers: int) -> None:
    started = datetime.now(timezone.utc)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    execution_lock = validate_execution_lock()
    write_json(LOOP_ROOT / "execution_lock_validation.json", execution_lock)
    if not execution_lock["passed"]:
        raise RuntimeError("execution lock validation failed")
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    labels, diagnostics, replay, execution_detail = execute_labels(manifest, workers)
    labels, historical_fresh_validation = replace_historical_with_frozen(labels)
    frozen_replay = frozen_input_replay(labels)
    all_replay = bool(
        replay["exactReplayPassed"].all()
        and historical_fresh_validation["passed"].all()
        and frozen_replay["passed"].all()
    )
    if not all_replay:
        replay.to_parquet(LOOP_ROOT / "label_replay_evidence.parquet", index=False)
        frozen_replay.to_parquet(LOOP_ROOT / "frozen_label_replay_evidence.parquet", index=False)
        raise RuntimeError("exact label replay failed; loop failed closed")
    fingerprints = build_fingerprints(labels, diagnostics)
    episodes = episode_table(labels)
    aggregate, summary_long = aggregate_fingerprints(fingerprints)
    comparisons, bootstrap_replicates, loo = paper_comparisons(fingerprints, aggregate)
    overlap, overlap_summary = label_overlap(labels)
    cross_candidate = cross_candidate_agreement(fingerprints)
    classification = classify(aggregate, comparisons, replay, frozen_replay, overlap_summary)
    labels.to_parquet(LOOP_ROOT / "label_values.parquet", index=False, compression="zstd")
    diagnostics.to_parquet(LOOP_ROOT / "label_fit_diagnostics.parquet", index=False)
    replay.to_parquet(LOOP_ROOT / "label_replay_evidence.parquet", index=False)
    historical_fresh_validation.to_parquet(
        LOOP_ROOT / "historical_fresh_replay_evidence.parquet", index=False
    )
    frozen_replay.to_parquet(LOOP_ROOT / "frozen_label_replay_evidence.parquet", index=False)
    fingerprints.to_parquet(LOOP_ROOT / "fingerprint_results.parquet", index=False)
    aggregate.to_csv(LOOP_ROOT / "fingerprint_aggregate.csv", index=False)
    summary_long.to_parquet(LOOP_ROOT / "fingerprint_summary.parquet", index=False)
    episodes.to_parquet(LOOP_ROOT / "episode_results.parquet", index=False)
    fingerprints[
        [
            "candidateId",
            "matrixIndex",
            "trajectoryId",
            "labelId",
            "cutoffCount",
            "cutoffIndex0",
            "isNonreplicatingAtCutoff",
            "noReplicatorObservedThroughCutoff",
            "firstOnsetRawIndex0",
            "firstOnsetNormalized",
        ]
    ].to_parquet(LOOP_ROOT / "cutoff_results.parquet", index=False)
    fingerprints[
        [
            "candidateId",
            "matrixIndex",
            "trajectoryId",
            "labelId",
            "globalReference",
            "referenceSize",
            "postFissionReplicatorCount",
            "postFissionReplicatorFraction",
            "postFissionEpisodeCount",
            "sameReferenceReentryCount",
            "sameReferenceTemporalSpanNormalized",
        ]
    ].to_parquet(LOOP_ROOT / "recurrence_results.parquet", index=False)
    comparisons.to_csv(LOOP_ROOT / "paper_fingerprint_comparison.csv", index=False)
    bootstrap_replicates.to_parquet(LOOP_ROOT / "paper_distance_bootstrap.parquet", index=False)
    loo.to_parquet(LOOP_ROOT / "leave_one_out_robustness.parquet", index=False)
    overlap.to_parquet(LOOP_ROOT / "label_overlap_with_adjacent_h.parquet", index=False)
    overlap_summary.to_csv(LOOP_ROOT / "label_overlap_summary.csv", index=False)
    cross_candidate.to_csv(LOOP_ROOT / "cross_candidate_agreement.csv", index=False)
    execution = replay.groupby(["candidateId", "matrixIndex"], as_index=False).agg(
        labelCount=("labelId", "size"), exactReplayPassed=("exactReplayPassed", "all")
    )
    execution["status"] = np.where(execution["exactReplayPassed"], "COMPLETE", "FAILED_CLOSED")
    execution.to_parquet(LOOP_ROOT / "execution_status.parquet", index=False)
    robustness = pd.concat(
        [
            comparisons.assign(robustnessType="PAPER_DISTANCE_BOOTSTRAP_AND_LOO"),
            overlap_summary.assign(robustnessType="ADJACENT_H_OVERLAP"),
            cross_candidate.assign(robustnessType="CROSS_CANDIDATE_AGREEMENT"),
        ],
        ignore_index=True,
        sort=False,
    )
    robustness.to_parquet(LOOP_ROOT / "robustness_results.parquet", index=False)
    write_json(LOOP_ROOT / "classification.json", classification)
    failures = pd.read_csv(LOOP_ROOT / "failure_ledger.csv")
    if len(failures):
        raise RuntimeError("unexpected pre-existing L02 failure")
    cpu_hours = (
        time.process_time() - started_cpu
        + float(execution_detail["cpuSeconds"].sum())
    ) / 3600
    cache_files = list(TRAJECTORY_CACHE.rglob("*.parquet"))
    wall_hours = (time.perf_counter() - started_wall) / 3600
    runtime = {
        "schema": "eidosoma.e01.s19_l02_runtime_manifest.v1",
        "loopId": LOOP_ID,
        "startedUtc": started.isoformat(),
        "completedUtc": datetime.now(timezone.utc).isoformat(),
        "repositoryCommit": execution_lock["repositoryHead"],
        "workers": workers,
        "threadsPerWorker": 1,
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "scientificCpuHours": cpu_hours,
        "wallHours": wall_hours,
        "gpuHours": 0.0,
        "gpuUsed": False,
        "trajectoryCount": 200,
        "labelTrajectoryFits": 1600,
        "bootstrapReplicatesPerComparison": BOOTSTRAP_REPLICATES,
        "cacheParquetCount": len(cache_files),
        "newGardTrajectoryCount": 0,
        "newPhiRLOrEmergenceCount": 0,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "ceilings": {"cpuHours": 48, "gpuHours": 0, "wallHours": 8},
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)
    retained = directory_bytes(LOOP_ROOT)
    temporary = directory_bytes(CACHE_ROOT)
    storage = {
        "schema": "eidosoma.e01.s19_l02_storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiB": retained / 2**30,
        "retainedCeilingGiB": 10,
        "temporaryBytes": temporary,
        "temporaryGiB": temporary / 2**30,
        "temporaryCeilingGiB": 25,
        "passed": retained <= 10 * 2**30 and temporary <= 25 * 2**30,
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    validation = {
        "schema": "eidosoma.e01.s19_l02_regeneration_validation.v1",
        "validationResult": "PASS_ALL_LOCK_REPLAY_IMMUTABILITY_STORAGE_AND_REGENERATION_CHECKS",
        "executionLock": "PASS" if execution_lock["passed"] else "FAIL",
        "immutablePrior": "PASS",
        "preanalysisReplay": "PASS",
        "allLabelReplay": "PASS" if replay["exactReplayPassed"].all() else "FAIL",
        "freshHistoricalTechnique1Replay": "PASS" if historical_fresh_validation["passed"].all() else "FAIL",
        "frozenInputReplay": "PASS" if frozen_replay["passed"].all() else "FAIL",
        "labelFamilyCount": int(labels["labelId"].nunique()),
        "candidateCount": int(labels["candidateId"].nunique()),
        "matrixCountPerCandidate": {
            str(key): int(value)
            for key, value in labels.groupby("candidateId")["matrixIndex"].nunique().items()
        },
        "fingerprintRowCount": len(fingerprints),
        "promotionLimitPassed": classification["promotedLeadCount"] <= 2,
        "storagePassed": storage["passed"],
        "newTrajectoryCount": 0,
        "newPhiRLOrEmergenceCount": 0,
        "L01ScientificValuesReadOrReinterpreted": False,
        "passed": bool(
            execution_lock["passed"]
            and replay["exactReplayPassed"].all()
            and historical_fresh_validation["passed"].all()
            and frozen_replay["passed"].all()
            and labels["labelId"].nunique() == 4
            and labels["candidateId"].nunique() == 2
            and len(fingerprints) == 800
            and classification["promotedLeadCount"] <= 2
            and storage["passed"]
        ),
    }
    provisional_artifacts = [str(path) for path in sorted(LOOP_ROOT.iterdir()) if path.is_file()]
    report, decision = build_reports(aggregate, comparisons, classification, validation, provisional_artifacts, runtime)
    report_second, decision_second = build_reports(aggregate, comparisons, classification, validation, provisional_artifacts, runtime)
    validation["reportDeterministicRegenerationPassed"] = report == report_second and decision == decision_second
    validation["passed"] &= validation["reportDeterministicRegenerationPassed"]
    write_json(LOOP_ROOT / "regeneration_validation.json", validation)
    (LOOP_ROOT / "S19_L02_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(decision, encoding="utf-8")
    append_postloop_ledger(classification, aggregate, datetime.now(timezone.utc).isoformat())
    current_artifacts = [str(path) for path in sorted(LOOP_ROOT.iterdir()) if path.is_file()]
    update_root_state(classification, report, current_artifacts)
    required = [
        "preregistration.yaml",
        "method_lock.json",
        "label_registry.yaml",
        "label_registry.parquet",
        "seed_manifest.parquet",
        "input_manifest.json",
        "source_snapshot_manifest.json",
        "preanalysis_replay_validation.json",
        "preanalysis_replay_evidence.parquet",
        "label_values.parquet",
        "fingerprint_results.parquet",
        "fingerprint_aggregate.csv",
        "fingerprint_summary.parquet",
        "episode_results.parquet",
        "cutoff_results.parquet",
        "recurrence_results.parquet",
        "paper_fingerprint_comparison.csv",
        "paper_distance_bootstrap.parquet",
        "leave_one_out_robustness.parquet",
        "label_replay_evidence.parquet",
        "frozen_label_replay_evidence.parquet",
        "historical_fresh_replay_evidence.parquet",
        "robustness_results.parquet",
        "failure_ledger.csv",
        "runtime_manifest.json",
        "storage_validation.json",
        "regeneration_validation.json",
        "classification.json",
        "loop_decision_summary.md",
        "S19_L02_FULL_RESULTS.md",
        "research_step_full_results.md",
    ]
    manifest_payload = artifact_manifest(
        LOOP_ROOT, required, "eidosoma.e01.s19_l02_artifact_manifest.v1"
    )
    write_json(LOOP_ROOT / "artifact_manifest.json", manifest_payload)
    if not manifest_payload["passed"] or not validation["passed"]:
        raise RuntimeError("final artifact validation failed")
    root_payload = root_manifest()
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", root_payload)
    if not root_payload["passed"]:
        raise RuntimeError("root S19 manifest validation failed")
    print(
        canonical_json(
            {
                "success": True,
                "loopId": LOOP_ID,
                "labelRows": len(labels),
                "fingerprintRows": len(fingerprints),
                "promotedLeadIds": classification["promotedLeadIds"],
                "recommendedNextAction": classification["defaultNextRecommendation"],
                "validation": validation["validationResult"],
            }
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be between 1 and 8")
    main(args.workers)
