#!/usr/bin/env python3
"""Finalize the S13 schema-gate failure without scientific aggregation or repair."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import yaml

from e01_confirmed_timebase_scaleup.core import (
    ANALYSIS_ROOT_SEED_HEX,
    CANDIDATE_IDS,
    RESEARCH_STEP_ID,
    VERSION,
    analysis_seed_material,
    derive_analysis_seed,
)
from e01_latent_timebase.core import derive_seed as derive_simulation_seed

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
ROOT = ARTIFACTS / "research_steps/S13"
CACHE = Path("/cache/e01_s13")
SOURCE_CACHE = CACHE / "source_results"
CONFIG = REPO / "configs/e01/s13_confirmed_timebase_baseline_scaleup_preregistration.yaml"
SCHEMAS = REPO / "configs/e01/s12g_output_schemas.json"
FAILURE_TOKEN = "SOURCE_LABEL_PARQUET_SCHEMA_MISMATCH"
FAILURE_SUMMARY = (
    "Per-task source calculations completed, but Arrow aggregation rejected divergent "
    "physical types for optional all-null label fields (null versus string/double)."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def validate_prior() -> dict[str, Any]:
    baseline = json.loads((ROOT / "immutable_prior_baseline.json").read_text())
    changed = []
    for item in baseline["files"]:
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        if actual != item["sha256"]:
            changed.append(
                {
                    "path": str(path),
                    "expectedSha256": item["sha256"],
                    "actualSha256": actual,
                }
            )
    payload = {
        "schema": "eidosoma.e01.s13_immutable_prior_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "fileCount": len(baseline["files"]),
        "changedCount": len(changed),
        "changed": changed,
        "passed": not changed,
    }
    write_json(ROOT / "immutable_prior_validation.json", payload)
    return payload


def cache_and_task_manifest() -> tuple[pd.DataFrame, dict[str, Any]]:
    task_rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate in CANDIDATE_IDS:
        for matrix_index in range(100):
            task_root = SOURCE_CACHE / candidate / f"M{matrix_index:02d}"
            completion = task_root / "completion.json"
            if not completion.is_file():
                missing.append(f"{candidate}/M{matrix_index:02d}")
                continue
            record = json.loads(completion.read_text())
            task_rows.append(record)
            for path in sorted(task_root.iterdir()):
                if path.is_file():
                    files.append(
                        {
                            "candidateId": candidate,
                            "matrixIndex": matrix_index,
                            "relativePath": str(path.relative_to(CACHE)),
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
    frame = pd.DataFrame(task_rows).sort_values(
        ["candidateId", "matrixIndex"]
    ).reset_index(drop=True)
    frame.to_parquet(
        ROOT / "source_task_validation.parquet", index=False, compression="zstd"
    )
    payload = {
        "schema": "eidosoma.e01.s13_source_cache_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "completeTaskCount": len(frame),
        "missingTasks": missing,
        "completeByCandidate": {
            candidate: int((frame["candidateId"] == candidate).sum())
            for candidate in CANDIDATE_IDS
        },
        "allFullReplayPassed": bool(frame["fullReplayAllPassed"].astype(bool).all()),
        "allPrefixReplayPassed": bool(
            frame["prefixReplayAllPassed"].astype(bool).all()
        ),
        "allSuffixPassed": bool(frame["futureSuffixAllPassed"].astype(bool).all()),
        "taskFailureRows": int(frame["failureRows"].sum()),
        "cacheFileCount": len(files),
        "cacheBytes": sum(item["bytes"] for item in files),
        "cacheFiles": files,
        "scientificAggregationPerformed": False,
        "candidateStatisticsComputed": False,
        "passed": bool(
            len(frame) == 200
            and not missing
            and frame["fullReplayAllPassed"].astype(bool).all()
            and frame["prefixReplayAllPassed"].astype(bool).all()
            and frame["futureSuffixAllPassed"].astype(bool).all()
            and int(frame["failureRows"].sum()) == 0
        ),
    }
    write_json(ROOT / "source_cache_manifest.json", payload)
    return frame, payload


def schema_diagnostics() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    table_names = (
        "labels.parquet",
        "preprocessing.parquet",
        "full.parquet",
        "prefix.parquet",
        "partition.parquet",
        "diagnostic.parquet",
        "suffix.parquet",
        "seeds.parquet",
        "failures.parquet",
    )
    for candidate in CANDIDATE_IDS:
        for matrix_index in range(100):
            task_root = SOURCE_CACHE / candidate / f"M{matrix_index:02d}"
            schemas: dict[str, Any] = {}
            for name in table_names:
                schema = pq.read_schema(task_root / name)
                signature = [(field.name, str(field.type)) for field in schema]
                schemas[name] = {
                    "sha256": sha256_json(signature),
                    "signature": signature,
                }
            label_types = dict(schemas["labels.parquet"]["signature"])
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": matrix_index,
                    "taskId": f"{candidate}/M{matrix_index:02d}",
                    "labelsSchemaSha256": schemas["labels.parquet"]["sha256"],
                    "clusterIdType": label_types["clusterId"],
                    "referenceObservationIdType": label_types[
                        "referenceObservationId"
                    ],
                    "metricToReferenceType": label_types["metricToReference"],
                    **{
                        f"{name.removesuffix('.parquet')}SchemaSha256": item["sha256"]
                        for name, item in schemas.items()
                    },
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["candidateId", "matrixIndex"]
    ).reset_index(drop=True)
    frame.to_csv(ROOT / "source_schema_diagnostics.csv", index=False, lineterminator="\n")
    counts = Counter(frame["labelsSchemaSha256"])
    variants = []
    for fingerprint, count in sorted(counts.items()):
        representative = frame[frame["labelsSchemaSha256"] == fingerprint].iloc[0]
        variants.append(
            {
                "labelsSchemaSha256": fingerprint,
                "taskCount": count,
                "representativeTaskId": representative["taskId"],
                "clusterIdType": representative["clusterIdType"],
                "referenceObservationIdType": representative[
                    "referenceObservationIdType"
                ],
                "metricToReferenceType": representative["metricToReferenceType"],
            }
        )
    payload = {
        "schema": "eidosoma.e01.s13_source_schema_failure_diagnostics.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "taskCount": len(frame),
        "labelSchemaVariantCount": len(variants),
        "labelSchemaVariants": variants,
        "failureToken": FAILURE_TOKEN,
        "failureSummary": FAILURE_SUMMARY,
        "schemaAdapterApplied": False,
        "sourceCalculationRerun": False,
        "candidateStatisticOpened": False,
        "globalSchemaGatePassed": False,
        "terminalStopApplied": True,
    }
    write_json(ROOT / "source_schema_failure_diagnostics.json", payload)
    return frame, payload


def simulation_seed_row(seed: Any, candidate_id: str | None) -> dict[str, Any]:
    shared = seed.purpose in {"catalytic_matrix", "initial_state"}
    identity_candidate = "SHARED" if shared else candidate_id
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "streamDomain": "simulation",
        "streamId": (
            f"S13::SIM::{seed.purpose}::M{int(seed.matrix_index):03d}::"
            f"{identity_candidate}"
        ),
        "purpose": seed.purpose,
        "candidateId": None if shared else candidate_id,
        "matrixIndex": int(seed.matrix_index),
        "implementationId": None,
        "temporalModeId": None,
        "endpointGeneration": None,
        "derivedSeed": str(seed.derived_seed),
        "seedMaterialSha256": seed.seed_material_sha256,
        "rootHex": seed.root_sha256,
        "bitGenerator": "PCG64DXSM",
        "sharedAcrossCandidates": shared,
        "executionStatus": "EXECUTED",
    }


def analysis_seed_row(
    *,
    domain: str,
    stream_id: str,
    purpose: str,
    identity: tuple[Any, ...],
    candidate_id: str,
    matrix_index: int,
    implementation_id: str,
    temporal_mode_id: str,
    endpoint_generation: int | None,
) -> dict[str, Any]:
    material = analysis_seed_material(*identity)
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "streamDomain": domain,
        "streamId": stream_id,
        "purpose": purpose,
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "implementationId": implementation_id,
        "temporalModeId": temporal_mode_id,
        "endpointGeneration": endpoint_generation,
        "derivedSeed": str(derive_analysis_seed(*identity)),
        "seedMaterialSha256": hashlib.sha256(material).hexdigest(),
        "rootHex": ANALYSIS_ROOT_SEED_HEX,
        "bitGenerator": (
            "MT19937_via_numpy_RandomState"
            if domain == "suffix"
            else "source_wrapper_seed32"
        ),
        "sharedAcrossCandidates": False,
        "executionStatus": "EXECUTED",
    }


def seed_manifest() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase = "s13_heldout_scaleup"
    for matrix_index in range(100):
        for purpose in ("catalytic_matrix", "initial_state"):
            rows.append(
                simulation_seed_row(
                    derive_simulation_seed(
                        ANALYSIS_ROOT_SEED_HEX, phase, purpose, matrix_index
                    ),
                    None,
                )
            )
        for candidate in CANDIDATE_IDS:
            for purpose in (
                "poisson_update",
                "overshoot_trim",
                "fission",
                "daughter_selection",
            ):
                rows.append(
                    simulation_seed_row(
                        derive_simulation_seed(
                            ANALYSIS_ROOT_SEED_HEX,
                            phase,
                            purpose,
                            matrix_index,
                            candidate,
                        ),
                        candidate,
                    )
                )
    source_seed_match = True
    for candidate in CANDIDATE_IDS:
        for matrix_index in range(100):
            task_root = SOURCE_CACHE / candidate / f"M{matrix_index:02d}"
            source = pd.read_parquet(task_root / "seeds.parquet")
            for item in source.to_dict("records"):
                endpoint = (
                    None
                    if pd.isna(item["endpointGeneration"])
                    else int(item["endpointGeneration"])
                )
                temporal = "FULL" if endpoint is None else "PREFIX_ENDPOINT"
                terminal: str | int = "NONE" if endpoint is None else endpoint
                identity = (
                    str(item["purpose"]),
                    candidate,
                    matrix_index,
                    str(item["implementationId"]),
                    temporal,
                    terminal,
                )
                expected = derive_analysis_seed(*identity)
                source_seed_match &= int(item["seed"]) == expected
                rows.append(
                    analysis_seed_row(
                        domain="source",
                        stream_id=str(item["streamId"]),
                        purpose=str(item["purpose"]),
                        identity=identity,
                        candidate_id=candidate,
                        matrix_index=matrix_index,
                        implementation_id=str(item["implementationId"]),
                        temporal_mode_id=str(item["temporalModeId"]),
                        endpoint_generation=endpoint,
                    )
                )
            prefix = pd.read_parquet(
                task_root / "prefix.parquet",
                columns=[
                    "implementationId",
                    "generation",
                    "priorLockedClockTransitions",
                ],
            )
            prefix = prefix[prefix["priorLockedClockTransitions"] >= 256]
            for item in prefix.to_dict("records"):
                for purpose in (
                    "suffix_deterministic_shuffle",
                    "suffix_domain_separated_replacement",
                ):
                    generation = int(item["generation"])
                    implementation = str(item["implementationId"])
                    identity = (
                        purpose,
                        candidate,
                        matrix_index,
                        implementation,
                        generation,
                    )
                    rows.append(
                        analysis_seed_row(
                            domain="suffix",
                            stream_id=(
                                f"S13::SUFFIX::{purpose}::{candidate}::"
                                f"M{matrix_index:03d}::{implementation}::G{generation:03d}"
                            ),
                            purpose=purpose,
                            identity=identity,
                            candidate_id=candidate,
                            matrix_index=matrix_index,
                            implementation_id=implementation,
                            temporal_mode_id="PREFIX_ENDPOINT",
                            endpoint_generation=generation,
                        )
                    )
    frame = pd.DataFrame(rows)
    frame.to_parquet(ROOT / "seed_manifest.parquet", index=False, compression="zstd")
    prior_streams: set[str] = set()
    prior_materials: set[str] = set()
    for path in sorted((ARTIFACTS / "research_steps").glob("S*/**/*seed*.parquet")):
        if "/S13/" in str(path):
            continue
        prior = pd.read_parquet(path)
        for column in prior.columns:
            lowered = column.lower()
            values = {str(value) for value in prior[column].dropna().tolist()}
            if lowered in {"streamid", "stream_id", "identity"}:
                prior_streams.update(values)
            if lowered in {"seedmaterialsha256", "seed_material_sha256"}:
                prior_materials.update(values)
    stream_overlap = sorted(set(frame["streamId"]) & prior_streams)
    material_overlap = sorted(set(frame["seedMaterialSha256"]) & prior_materials)
    payload = {
        "schema": "eidosoma.e01.s13_seed_firewall.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "executedSeedRows": len(frame),
        "uniqueStreamIds": int(frame["streamId"].nunique()),
        "uniqueSeedMaterials": int(frame["seedMaterialSha256"].nunique()),
        "sourceSeedValuesMatchFrozenDerivation": source_seed_match,
        "streamIdentityOverlapCount": len(stream_overlap),
        "seedMaterialOverlapCount": len(material_overlap),
        "streamIdentityOverlap": stream_overlap,
        "seedMaterialOverlap": material_overlap,
        "statisticsSeedsExecuted": 0,
        "passed": bool(
            len(frame) == frame["streamId"].nunique()
            and len(frame) == frame["seedMaterialSha256"].nunique()
            and source_seed_match
            and not stream_overlap
            and not material_overlap
        ),
    }
    write_json(ROOT / "seed_firewall.json", payload)
    return frame, payload


def create_suppressed_outputs(reason: str) -> None:
    contract = json.loads(SCHEMAS.read_text())["tables"]
    for filename, columns in contract.items():
        path = ROOT / filename
        if path.is_file():
            continue
        frame = pd.DataFrame(columns=columns)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False, compression="zstd")
        else:
            frame.to_csv(path, index=False, lineterminator="\n")
    pd.DataFrame(
        columns=[
            "candidateId",
            "trajectoryId",
            "matrixIndex",
            "implementationId",
            "generation",
            "endpointRawObservationIndex",
            "rawObservationIndex",
        ]
    ).to_parquet(
        ROOT / "prefix_statistical_view_index.parquet",
        index=False,
        compression="zstd",
    )
    pd.DataFrame(
        columns=[
            "candidateId",
            "matrixIndex",
            "trajectoryId",
            "implementationId",
            "temporalMode",
            "estimand",
            "correlation",
            "ordinaryP",
        ]
    ).to_parquet(
        ROOT / "candidate_association_details.parquet",
        index=False,
        compression="zstd",
    )
    pd.DataFrame(
        columns=[
            "candidateId",
            "matrixIndex",
            "trajectoryId",
            "implementationId",
            "temporalMode",
            "meanDifference",
            "medianDifference",
        ]
    ).to_parquet(
        ROOT / "replicator_drift_details.parquet",
        index=False,
        compression="zstd",
    )
    write_json(
        ROOT / "adapter_validation.json",
        {
            "schema": "eidosoma.e01.s13_adapter_validation.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "status": "NOT_REACHED_SOURCE_SCHEMA_GATE",
            "reason": reason,
            "adapterApplied": False,
            "passed": False,
        },
    )
    write_json(
        ROOT / "statistics_replay_validation.json",
        {
            "schema": "eidosoma.e01.s13_statistics_replay_validation.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "status": "NOT_REACHED_SOURCE_SCHEMA_GATE",
            "statisticsExecutions": 0,
            "reason": reason,
            "passed": False,
        },
    )
    figure_root = ROOT / "figures"
    figure_root.mkdir(exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text())
    for relative in config["artifacts"]["figures"]:
        path = ROOT / relative
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.axis("off")
        axis.text(
            0.5,
            0.58,
            "S13 STOPPED FAIL-CLOSED",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
        )
        axis.text(
            0.5,
            0.40,
            "Source label-table schema mismatch\nNo scientific statistic computed",
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def artifact_schema_validation() -> dict[str, Any]:
    contract = json.loads(SCHEMAS.read_text())["tables"]
    rows = []
    for filename, columns in contract.items():
        path = ROOT / filename
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        missing = [column for column in columns if column not in frame.columns]
        rows.append(
            {
                "path": filename,
                "exists": path.is_file(),
                "rowCount": len(frame),
                "missingColumns": missing,
                "artifactSchemaPresent": not missing,
                "scientificStatus": (
                    "PARTIAL_UNPROMOTED_SCHEMA_FAILURE"
                    if filename == "label_values.parquet" and len(frame)
                    else "SUPPRESSED_NOT_REACHED"
                ),
            }
        )
    payload = {
        "schema": "eidosoma.e01.s13_artifact_schema_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "tables": rows,
        "artifactSchemasPresent": all(not row["missingColumns"] for row in rows),
        "sourceAggregationSchemaGatePassed": False,
        "scientificTablesEligible": False,
        "passed": False,
        "reason": FAILURE_TOKEN,
    }
    write_json(ROOT / "schema_validation.json", payload)
    return payload


def artifact_manifest(required: list[str]) -> dict[str, Any]:
    entries = [
        {
            "relativePath": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    present = {item["relativePath"] for item in entries}
    missing = [
        item
        for item in required
        if item != "artifact_manifest.json" and item not in present
    ]
    total = sum(item["bytes"] for item in entries)
    payload = {
        "schema": "eidosoma.e01.s13_artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifacts": entries,
        "artifactCountExcludingSelf": len(entries),
        "totalBytesExcludingSelf": total,
        "requiredMissing": missing,
        "under30GiB": total <= 30 * 1024**3,
        "scientificValidationPassed": False,
        "handoffCompletenessPassed": not missing and total <= 30 * 1024**3,
        "passed": not missing and total <= 30 * 1024**3,
    }
    write_json(ROOT / "artifact_manifest.json", payload)
    return payload


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text())
    method_lock = json.loads((ROOT / "method_lock.json").read_text())
    compute = json.loads((ROOT / "compute_ledger.json").read_text())
    benchmark = json.loads((ROOT / "runtime_benchmark.json").read_text())
    prior = validate_prior()
    tasks, cache = cache_and_task_manifest()
    schema_frame, schema_failure = schema_diagnostics()
    seeds, firewall = seed_manifest()
    create_suppressed_outputs(FAILURE_SUMMARY)
    schema = artifact_schema_validation()

    start = datetime.fromtimestamp(
        (ROOT / "implementation_lock.json").stat().st_mtime, timezone.utc
    )
    stop = datetime.fromtimestamp((ROOT / "status.json").stat().st_mtime, timezone.utc)
    worker_cpu = float(tasks["cpuSeconds"].sum()) / 3600
    worker_wall = float(tasks["wallSeconds"].sum()) / 3600
    approximate_wall = (stop - start).total_seconds() / 3600
    observed_cumulative = (
        float(compute["priorCpuEnvelopeRoundedForDecision"]) + worker_cpu + 0.25
    )
    runtime = {
        "schema": "eidosoma.e01.s13_runtime_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "status": "STOPPED_FAIL_CLOSED",
        "startedAtUtcFromImplementationLockMtime": start.isoformat(),
        "stoppedAtUtcFromTerminalStatusMtime": stop.isoformat(),
        "approximateWallHours": approximate_wall,
        "summedSourceWorkerCpuHours": worker_cpu,
        "summedSourceWorkerWallHours": worker_wall,
        "simulationCpuHoursRetainedDirectly": None,
        "simulationCpuAccountingPolicy": (
            "not retained after terminal exception; covered by 0.25-hour orchestration/"
            "simulation envelope in cumulative audit"
        ),
        "observedS13CpuEnvelopeHours": worker_cpu + 0.25,
        "observedCumulativeE01CpuEnvelopeHours": observed_cumulative,
        "cumulativeCpuCeilingHours": 250.0,
        "gpuHours": 0.0,
        "cumulativeGpuEnvelopeHours": float(compute["priorGpuEnvelopeHours"]),
        "cumulativeGpuCeilingHours": 80.0,
        "workers": 6,
        "threadEnvironment": {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "cpuFloat64Authoritative": True,
        "hardCeilingExceeded": bool(
            approximate_wall > 72
            or observed_cumulative > 250
            or float(compute["priorGpuEnvelopeHours"]) > 80
        ),
        "terminalReason": FAILURE_TOKEN,
        "passed": approximate_wall <= 72 and observed_cumulative <= 250,
    }
    write_json(ROOT / "runtime_manifest.json", runtime)
    cache_bytes = sum(path.stat().st_size for path in CACHE.rglob("*") if path.is_file())
    artifact_bytes = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s13_storage_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "cacheBytes": cache_bytes,
        "artifactBytesBeforeFinalManifest": artifact_bytes,
        "retainedArtifactGiBCeiling": 30.0,
        "passed": artifact_bytes <= 30 * 1024**3,
    }
    write_json(ROOT / "storage_validation.json", storage)
    provenance = {
        "schema": "eidosoma.e01.s13_provenance_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "designCommit": method_lock["designCommit"],
        "reportingCommit": git("rev-parse", "HEAD"),
        "reportingOnlyFinalizer": "scripts/e01/finalize_s13_fail_closed.py",
        "reportingFinalizerChangedScientificMethod": False,
        "branch": git("branch", "--show-current"),
        "sourceCommits": {
            "IIGR": "7c1c22fe39f539d4a453135476f1f0dd5a6b45f7",
            "PhiRL": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373",
            "historicalGARD": "86dff6320d5ae91b4e831471079ff46749b14df9",
        },
        "candidateIds": list(CANDIDATE_IDS),
        "candidate1Executed": False,
        "retainedPrimaryTrajectoryCount": 200,
        "validationReplayExecutionCount": 200,
        "sourceTaskCount": 200,
        "candidateStatisticsComputed": False,
        "sourceValuesUsedScientifically": False,
        "passed": True,
    }
    write_json(ROOT / "provenance_manifest.json", provenance)
    access = json.loads((ROOT / "scope_access_ledger.json").read_text())
    access["events"].append(
        {
            "stage": "STOP_FAIL_CLOSED_AT_SOURCE_SCHEMA_AGGREGATION",
            "simulationOutcomeOpened": True,
            "sourceTaskValuesComputedToCache": True,
            "partialLabelTableWrittenBeforeFailure": True,
            "candidateStatisticOpened": False,
            "associationOrAdjudicationComputed": False,
            "schemaAdapterApplied": False,
            "sourceRerun": False,
            "predictionOrInterventionAccess": False,
            "s14ThroughS18Access": False,
            "status": "FAIL_CLOSED",
            "reason": FAILURE_TOKEN,
        }
    )
    access["success"] = False
    write_json(ROOT / "scope_access_ledger.json", access)
    pd.DataFrame(
        [
            {
                "failureId": "S13-TERMINAL-SOURCE-SCHEMA",
                "stage": "source_table_aggregation",
                "candidateId": None,
                "trajectoryId": None,
                "implementationId": None,
                "temporalModeId": None,
                "endpointGeneration": None,
                "severity": "FATAL",
                "status": "S13_VALIDATION_FAILED_CLOSED",
                "reason": FAILURE_SUMMARY,
                "gateImpact": "GLOBAL_FAIL_CLOSED_NO_SCIENTIFIC_ADJUDICATION",
                "repairAttempted": False,
            },
            {
                "failureId": "S13-LABEL-SCHEMA-VARIANTS",
                "stage": "source_schema_diagnostic",
                "candidateId": None,
                "trajectoryId": None,
                "implementationId": None,
                "temporalModeId": None,
                "endpointGeneration": None,
                "severity": "FATAL",
                "status": FAILURE_TOKEN,
                "reason": (
                    f"{schema_failure['labelSchemaVariantCount']} incompatible label "
                    "schema variants across 200 completed tasks; see diagnostics"
                ),
                "gateImpact": "NO_ADAPTER_NO_RERUN_NO_STATISTICS",
                "repairAttempted": False,
            },
        ]
    ).to_csv(ROOT / "failure_ledger.csv", index=False, lineterminator="\n")
    classification = {
        "schema": "eidosoma.e01.s13_classification.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "classification": "S13_VALIDATION_FAILED_CLOSED",
        "outcomeClass": "constraining/contradictory_operational",
        "scientificAssociationClassification": "NOT_EVALUATED",
        "candidateSpecificStatisticsComputed": False,
        "twoCandidateAdjudicationPerformed": False,
        "candidate1Excluded": True,
        "candidate1EvidenceStatusRetained": (
            "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED"
        ),
        "reason": FAILURE_TOKEN,
        "s14ThroughS18Status": "BLOCKED_PENDING_S13_HUMAN_REVIEW",
    }
    write_json(ROOT / "classification.json", classification)
    execution = {
        "schema": "eidosoma.e01.s13_execution_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "retainedTrajectoryCount": 200,
        "completeTrajectoryCount": 200,
        "trajectoryReplayPassCount": 200,
        "sharedIdentityCount": 100,
        "sourceTaskCount": len(tasks),
        "sourceTaskReplayAndSuffixPassCount": int(
            (
                tasks["fullReplayAllPassed"].astype(bool)
                & tasks["prefixReplayAllPassed"].astype(bool)
                & tasks["futureSuffixAllPassed"].astype(bool)
                & (tasks["failureRows"] == 0)
            ).sum()
        ),
        "sourceCacheManifestPassed": cache["passed"],
        "seedFirewallPassed": firewall["passed"],
        "priorImmutabilityPassed": prior["passed"],
        "runtimePassed": runtime["passed"],
        "storagePassed": storage["passed"],
        "sourceAggregationSchemaPassed": False,
        "candidateStatisticsComputed": False,
        "allValidationGatesPassed": False,
        "validationResult": (
            "FAIL_CLOSED_AT_SOURCE_LABEL_PARQUET_SCHEMA_GATE_AFTER_200_OF_200_"
            "TRAJECTORY_AND_SOURCE_TASK_COMPLETIONS"
        ),
    }
    write_json(ROOT / "execution_validation.json", execution)
    report = f"""# S13 Full Results: Confirmed Time-base Baseline Held-out Scale-up

## Top summary

- **Research step ID:** `{VERSION}` (S13).
- **Completion status:** `STOPPED_FAIL_CLOSED_AT_SOURCE_SCHEMA_AGGREGATION`; no candidate association statistic or two-candidate adjudication was computed.
- **Artifacts written:** Complete pre-outcome locks and compute audit; 200 trajectory identities and replay rows; 100 pairing rows; 200 source-task validations; complete cache/schema/seed/runtime/storage/provenance/immutability/failure evidence; schema-bearing suppressed downstream outputs; six stop-state figures; artifact hashes; and this canonical report.
- **Validation result:** `{execution['validationResult']}`. Simulation and task-level source replay/suffix gates passed, but the frozen global schema gate failed.
- **Outcome classification:** `S13_VALIDATION_FAILED_CLOSED` (constraining/contradictory operational evidence); the held-out association question is `NOT_EVALUATED`.
- **Caveats or blockers:** The mismatch concerns physical Parquet types for optional all-null label fields. An adapter or schema cast would be a repair after source outcomes existed and is forbidden. The source-informed pipeline still cannot identify the unavailable author implementation, and repeated E01 overrides remain a procedural-credibility caveat.
- **Lay summary:** The new simulations and all 200 expensive source calculations finished and replayed, but their per-trajectory label tables did not share one physical file schema. The preregistered rule required an immediate stop on any schema failure. Therefore none of the generated emergence values was turned into a correlation or scientific verdict.
- **Recommended next action:** Mandatory human review. Do not repair or rerun S13, and keep S14–S18, prediction, MLP work, interventions, estimator repair, report-bundle progression, E02, and every further scale-up blocked.

## Frozen question and scope

S13 asked whether S12J's near-zero/non-support result persists on 100 genuinely new catalytic matrices shared across the two S12FR-confirmed time-base candidates. Candidate 2 used fixed exposure 0.6031526490073492, first-daughter continuation, trimmed new entrants, and C1. Candidate 3 used fixed exposure 0.5613315384859516, random-nonempty continuation, trimmed new entrants, and C1. Candidate 1 was not run and remains `HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED`.

A positive held-out result required both confirmed candidates to pass the unchanged retrospective and prospective rules after scaling only the preregistered count gates from 32 to 100 matrices. No positive/negative scientific classification is possible because aggregation stopped before any candidate statistic.

## Inputs and provenance

- Pushed outcome-blind design commit: `{method_lock['designCommit']}`.
- Reporting-only failure finalizer commit: `{provenance['reportingCommit']}`; it did not cast, adapt, concatenate, or analyze source values.
- New S13 seed-root ID: `E01-S13-HELDOUT-ROOT-v1.0.0`; {len(seeds):,} executed stream identities were reconstructed and audited.
- IIGR commit `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`; PhiRL commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; historical GARD commit `86dff6320d5ae91b4e831471079ff46749b14df9`.
- S12C source equivalence remained 14/14 and S12D source-emergence identity remained 40/40 before execution.
- All {prior['fileCount']} frozen S01–S12J artifact files retained their preregistered hashes.

## Detailed methods and commands

The simulator generated 100 shared catalytic matrices and 100 shared 40-distinct-singleton initial states from the new domain. Each matrix was run under both candidates for exactly 100 fissions. A second same-seed execution validated every trajectory using the S12FR exact comparator. The retained C1 state sequence recorded the initial state, every Poisson batch update, and every selected post-fission daughter.

Each retained trajectory then ran the frozen S12J label and source contract: historical H>0.9 primary label, past-only cosine secondary label, additive-0.5 closure, full CLR, removal of original component 100, IIGR source-defined emergence primary, PhiRL emergence robustness, and corrected local Phi-r comparator. Full fits were retrospective. Prefix fits were independently refit at post-fission endpoints after 256 C1 transitions, replayed exactly, and checked against suffix deletion, shuffle, and replacement. These per-task calculations were cached, not statistically aggregated.

```bash
PYTHONPATH=src python -m pytest -q tests/e01/test_s13_confirmed_timebase_scaleup.py tests/e01/test_s12g_frozen_timebase_ensemble.py tests/e01/test_s12j_aggregation_interface_repair_confirmation.py
python -m ruff check src/e01_confirmed_timebase_scaleup scripts/e01/freeze_s13_preregistration.py scripts/e01/run_s13_confirmed_timebase_scaleup.py tests/e01/test_s13_confirmed_timebase_scaleup.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s13_preregistration.py --record-commit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s13_confirmed_timebase_scaleup.py --workers 6
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/finalize_s13_fail_closed.py
```

## Results

- Retained primary trajectories: 200/200; complete to 100 fissions: 200/200.
- Exact simulator replay: 200/200; shared catalytic-matrix and initial-state identities: 100/100.
- Completed source tasks: {len(tasks)}/200; tasks with full replay, eligible-prefix replay, suffix flags, and zero failure rows: {execution['sourceTaskReplayAndSuffixPassCount']}/200.
- Candidate statistics, bootstrap, circular shift, block-aware inference, spike aggregation, metric-identity comparison, future-dependence summary, and final two-candidate gate: all `NOT_REACHED_SOURCE_SCHEMA_GATE`.
- No source value was used to select, eliminate, reweight, or reclassify a candidate.

## Terminal schema failure

The source worker writes one Parquet file per trajectory. Optional label columns become Arrow `null` when every value in a task is absent, but become `string` or `double` in tasks containing defined values. The completed campaign contained {schema_failure['labelSchemaVariantCount']} label-schema variants. The first global concatenation used one task schema and rejected another before a combined label table or any statistic existed. `source_schema_diagnostics.csv` preserves all 200 pair identities and table-schema fingerprints; `source_schema_failure_diagnostics.json` preserves representative types and counts.

No cast, alias, adapter, row deletion, schema normalization, cache reuse, source rerun, or partial-candidate analysis was attempted. The partial `label_values.parquet` written before Arrow raised is retained and explicitly marked `PARTIAL_UNPROMOTED_SCHEMA_FAILURE`; it is not scientific evidence. Every other downstream table is schema-bearing and empty.

## Validation

- Pre-simulation compute gate: PASS, projected cumulative {compute['projectedCumulativeCpuHours']:.3f}/250 CPU-hours and {compute['projectedCumulativeGpuHours']:.3f}/80 GPU-hours.
- Benchmark gate: PASS; projected cumulative {benchmark['projectedCumulativeCpuHours']:.3f} CPU-hours, {benchmark['projectedWallHoursAtSixWorkers']:.3f} wall-hours, and {benchmark['projectedRetainedGiB']:.3f} GiB.
- Prior immutability: PASS, {prior['fileCount']}/{prior['fileCount']} files unchanged.
- Seed firewall: {'PASS' if firewall['passed'] else 'FAIL'}, {len(seeds):,} unique executed streams, zero prior stream/material overlap, no statistics seed executed.
- Task-level replay/suffix evidence: PASS, 200/200 tasks and zero task failure rows.
- Global source schema: FAIL, {schema_failure['labelSchemaVariantCount']} label schema variants; terminal stop applied.
- Artifact table schemas: present, but scientific eligibility is false and the label table is partial/unpromoted.
- Runtime/storage: {'PASS' if runtime['passed'] and storage['passed'] else 'FAIL'} against hard ceilings; approximately {runtime['approximateWallHours']:.3f} wall-hours, {runtime['summedSourceWorkerCpuHours']:.3f} source-worker CPU-hours, zero GPU-hours, and {storage['cacheBytes'] / 1024**3:.3f} GiB cache.

## Caveats, blockers, and limitations

- This is an operational validation failure, not held-out positive or negative association evidence.
- A schema cast would likely be mechanically narrow, but it is still a prohibited repair after source values existed; S13 does not test or endorse one.
- The run extended a long E01 chain with repeated negative results and human overrides. Even a future positive result would require adjudication against that history.
- Full-trajectory values are retrospective and future-dependent. Fixed-window, pre-256-transition, early-warning, prediction, and causal-control claims remain unresolved and were not tested.
- Public IIGR/PhiRL behavior is source-informed and is not the unpublished author implementation or exact paper replication.

## Provenance and artifact contract

The design/method lock, prior hash baseline and postcheck, compute ledger, benchmark, source snapshot, 100 pairing identities, 200 trajectory hashes/replay rows, 200 source-task completion records, 2,000 cache-file hashes, all 200 source schema fingerprints, executed seed manifest/firewall, failure ledger, scope record, runtime/storage reports, suppression schemas, and collectible artifact hashes preserve the complete stop state. Large raw/source caches remain under `/cache/e01_s13/` and are represented by hashes.

## Recommended next action

Return for mandatory human review. The frozen no-repair rule has fired. Do not begin S14–S18, prediction, MLP work, interventions, estimator repair, report-bundle progression, E02, or another scale-up automatically.
"""
    (ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": VERSION,
        "stepNumber": "S13",
        "success": False,
        "status": "STOPPED_FAIL_CLOSED_AT_SOURCE_SCHEMA_AGGREGATION",
        "artifactsWritten": [],
        "validationResult": execution["validationResult"],
        "caveatsOrBlockers": [
            FAILURE_SUMMARY,
            "No schema adapter, source rerun, or scientific statistic is authorized.",
            "The held-out association question is NOT_EVALUATED.",
            "S14-S18 and all other downstream work remain blocked.",
        ],
        "recommendedNextAction": (
            "Mandatory human review; do not repair, rerun, or continue automatically."
        ),
        "outcomeClassification": "S13_VALIDATION_FAILED_CLOSED",
        "outcomeClass": "constraining/contradictory_operational",
        "scientificAssociationClassification": "NOT_EVALUATED",
        "s14ThroughS18Status": "BLOCKED_PENDING_S13_HUMAN_REVIEW",
    }
    write_json(ROOT / "status.json", status)
    manifest = artifact_manifest(
        [*config["artifacts"]["required"], *config["artifacts"]["figures"]]
    )
    status["artifactsWritten"] = [
        item["relativePath"] for item in manifest["artifacts"]
    ] + ["artifact_manifest.json"]
    write_json(ROOT / "status.json", status)
    manifest = artifact_manifest(
        [*config["artifacts"]["required"], *config["artifacts"]["figures"]]
    )
    if not (
        prior["passed"]
        and cache["passed"]
        and firewall["passed"]
        and runtime["passed"]
        and storage["passed"]
        and manifest["passed"]
        and len(schema_frame) == 200
        and not schema["sourceAggregationSchemaGatePassed"]
    ):
        raise RuntimeError("S13 fail-closed handoff validation failed")
    print(
        json.dumps(
            {
                "classification": classification["classification"],
                "scientificAssociationClassification": "NOT_EVALUATED",
                "completeTrajectories": 200,
                "completeSourceTasks": len(tasks),
                "schemaVariantCount": schema_failure["labelSchemaVariantCount"],
                "artifactManifestPassed": manifest["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
