#!/usr/bin/env python3
"""Prepare, fixture-test, benchmark, and freeze the pre-outcome S19-L09 lock."""

from __future__ import annotations

import hashlib
import json
import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import pickle
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_s19_recurring_attractor.core import (
    BOOTSTRAP_REPLICATES,
    EARLY_STOP_STREAK,
    K_VALUES,
    LABEL_IDS,
    MAX_ITERATIONS,
    MINIMUM_REFERENCE_MEMBERSHIP_VISITS,
    MINIMUM_VALID_CLUSTER_SIZE,
    RANDOM_REFERENCE_DRAWS,
    REPLICAS,
    R1_ID,
    R2_ID,
    ROOT_SEED_HEX,
    THRESHOLD,
    VERSION,
    array_sha256,
    close_rows,
    fit_r1_historical,
    fit_r2_euclidean,
    historical_h,
    historical_nondrift_technique1,
    label_against_reference,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L09"
CACHE_ROOT = Path("/cache/e01_s19_l09")
L08_CACHE = Path("/cache/e01_s19_l08/trajectories")
L08_ROOT = ARTIFACT_ROOT / "loops/L08"
CONFIG_PATH = REPO / "configs/e01/s19_l09_recurring_attractor.yaml"
CORE_PATH = REPO / "src/e01_s19_recurring_attractor/core.py"
RUNNER_PATH = REPO / "scripts/e01/run_s19_l09.py"
PREPARE_PATH = Path(__file__).resolve()
TEST_PATH = REPO / "tests/e01/test_s19_l09.py"
PAPER_MD = Path(
    "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
)
PAPER_PDF = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
PAPER_FIGURE_1 = Path(
    "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures/figure-01.png"
)
HISTORICAL_ROOT = Path("/cache/e01_s03/sources/gard-historical")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json_safe(value), sort_keys=False), encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def prior_roots() -> list[Path]:
    roots: list[Path] = []
    step_root = Path("/artifacts/research_steps")
    for path in sorted(step_root.iterdir()):
        if path.name != "S19":
            roots.append(path)
    for loop in ("L01", "L02", "L03", "L04", "L05", "L06", "L06R", "L07", "L08"):
        roots.append(ARTIFACT_ROOT / "loops" / loop)
    for bundle in (
        Path("/artifacts/E01_forensic_replication_bundle"),
        Path("/artifacts/E01_forensic_replication_artifact_v2"),
    ):
        if bundle.exists():
            roots.append(bundle)
    return roots


def immutable_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in prior_roots():
        if not root.exists():
            raise FileNotFoundError(root)
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in files:
            rows.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return rows


def fixture_values(center: int, count: int, width: int = 100) -> np.ndarray:
    values = np.zeros((count, width), dtype=np.float64)
    values[:, center] = 1.0
    values[:, (center + 1) % width] = 0.04
    for row in range(count):
        values[row, (center + 2 + row % 3) % width] = 0.002 * (row % 4)
    return close_rows(values)


def run_fixtures() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    source = np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=float)
    mask, angles, local = historical_nondrift_technique1(source)
    manual_angles = np.asarray([1.0, 1.0, 0.0, 1.0])
    manual_local = np.asarray([1.0, 0.5, 0.5, 1.0])
    rows.append(
        {
            "fixtureId": "F01_TGS_NONDRIFT_TECHNIQUE1",
            "pipelineId": R1_ID,
            "check": "source_equivalent_local_smoothing_and_strict_threshold",
            "passed": bool(
                np.array_equal(angles, manual_angles)
                and np.array_equal(local, manual_local)
                and np.array_equal(mask, manual_local > THRESHOLD)
            ),
            "details": "manual incoming/outgoing average including repeated endpoint pairs",
        }
    )

    two = np.vstack((fixture_values(1, 60), fixture_values(50, 40)))
    for pipeline_id, fitter in ((R1_ID, fit_r1_historical), (R2_ID, fit_r2_euclidean)):
        fit = fitter(two, "fixture-two-attractor")
        passed = bool(
            fit.status == "ELIGIBLE"
            and fit.second_cluster_id is not None
            and fit.cluster_sizes[fit.dominant_cluster_id] > fit.cluster_sizes[fit.second_cluster_id]
            and int(np.argmax(fit.dominant_centroid)) == 1
        )
        rows.append(
            {
                "fixtureId": "F02_TWO_ATTRACTOR_DOMINANCE",
                "pipelineId": pipeline_id,
                "check": "retain_two_clusters_select_more_recurrent",
                "passed": passed,
                "details": json.dumps(
                    {
                        "status": fit.status,
                        "selectedK": fit.selected_k,
                        "clusterSizes": fit.cluster_sizes,
                        "dominant": fit.dominant_cluster_id,
                        "second": fit.second_cluster_id,
                    },
                    sort_keys=True,
                ),
            }
        )

    drift = np.eye(100, dtype=np.float64)
    drift_r1 = fit_r1_historical(drift, "fixture-drift")
    drift_r2 = fit_r2_euclidean(drift, "fixture-drift")
    for pipeline_id, fit in ((R1_ID, drift_r1), (R2_ID, drift_r2)):
        rows.append(
            {
                "fixtureId": "F03_DRIFT_NO_DOMINANT_CLAIM",
                "pipelineId": pipeline_id,
                "check": "drifting_fixture_ineligible",
                "passed": bool(fit.status.startswith("INELIGIBLE_")),
                "details": fit.status,
            }
        )

    base = np.vstack((fixture_values(3, 63), fixture_values(55, 37)))
    permutation = np.roll(np.arange(100), 17)
    scaled = close_rows(base[:, permutation] * np.linspace(1, 5, len(base))[:, None])
    for pipeline_id, fitter in ((R1_ID, fit_r1_historical), (R2_ID, fit_r2_euclidean)):
        first = fitter(base, "fixture-invariance")
        replay = fitter(base, "fixture-invariance")
        transformed = fitter(scaled, "fixture-invariance")
        rows.extend(
            [
                {
                    "fixtureId": "F04_EXACT_SEED_REPLAY",
                    "pipelineId": pipeline_id,
                    "check": "exact_seed_replay",
                    "passed": bool(
                        first.status == replay.status
                        and first.selected_k == replay.selected_k
                        and np.array_equal(first.labels, replay.labels)
                        and np.array_equal(first.centroids, replay.centroids)
                    ),
                    "details": f"selectedK={first.selected_k}",
                },
                {
                    "fixtureId": "F05_FEATURE_PERMUTATION_SCALING_CLOSURE",
                    "pipelineId": pipeline_id,
                    "check": "permutation_and_positive_row_scaling_equivalence",
                    "passed": bool(
                        first.status == transformed.status == "ELIGIBLE"
                        and first.selected_k == transformed.selected_k
                        and np.array_equal(first.labels, transformed.labels)
                        and np.allclose(
                            first.dominant_centroid[permutation],
                            transformed.dominant_centroid,
                            atol=1e-12,
                            rtol=1e-12,
                        )
                    ),
                    "details": "feature permutation plus positive row scaling followed by closure",
                },
            ]
        )

    tie = np.vstack((fixture_values(2, 50), fixture_values(60, 50)))
    for pipeline_id, fitter in ((R1_ID, fit_r1_historical), (R2_ID, fit_r2_euclidean)):
        fit = fitter(tie, "fixture-tie")
        rows.append(
            {
                "fixtureId": "F06_DETERMINISTIC_TIE",
                "pipelineId": pipeline_id,
                "check": "equal_frequency_tie_is_deterministic",
                "passed": bool(
                    fit.status == "ELIGIBLE"
                    and fit.dominant_cluster_id == 0
                    and np.array_equal(
                        fit.labels, fitter(tie, "fixture-tie").labels
                    )
                ),
                "details": f"dominantClusterId={fit.dominant_cluster_id}",
            }
        )

    scores, labels = label_against_reference(
        np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float),
        np.asarray([1, 0, 0], dtype=float),
    )
    for pipeline_id in LABEL_IDS:
        rows.append(
            {
                "fixtureId": "F07_DIRECT_MOLECULAR_MEMBERSHIP",
                "pipelineId": pipeline_id,
                "check": "molecular_labels_not_interval_projection",
                "passed": bool(
                    np.array_equal(labels, [True, False, True])
                    and np.allclose(scores, [1, 0, 1])
                ),
                "details": "alternating molecular states remain alternating",
            }
        )

    frame = pd.DataFrame(rows)
    if not bool(frame["passed"].all()):
        failed = frame.loc[~frame["passed"]]
        raise RuntimeError(f"mandatory L09 fixtures failed: {failed.to_dict('records')}")
    return frame


def source_manifest() -> dict[str, Any]:
    source_files = [
        HISTORICAL_ROOT / name
        for name in (
            "tgs_nondrift.m",
            "tgs_H.m",
            "tgs_parameters_v10.m",
            "tgs_acluster.m",
            "tgs_kmeans.m",
            "tgs_carpet.m",
            "getcomposometime_v10.m",
            "biased_gard_v10.m",
            "README.txt",
        )
    ]
    references = [
        Path("/cache/e01_s19_l09/sources/PMC18166.html"),
        Path("/cache/e01_s19_l09/sources/PubMed_11735293.html"),
        Path("/cache/e01_s19_l09/sources/PubMed_11536890.html"),
        Path("/cache/e01_s19_l09/sources/ref64_GARD_domain.pdf"),
    ]
    retained = []
    for path in [PAPER_MD, PAPER_PDF, PAPER_FIGURE_1, *source_files, *references]:
        if not path.exists():
            raise FileNotFoundError(path)
        retained.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "redistribution": "REFERENCE_ONLY_NOT_COPIED_TO_ARTIFACTS"
                if path.parent != PAPER_MD.parent
                else "UPLOADED_INPUT_REFERENCE_ONLY",
            }
        )
    return {
        "schema": "eidosoma.e01.s19_l09.source_snapshot_manifest.v1",
        "capturedAtUtc": utc_now(),
        "historicalGard": {
            "repository": "https://github.com/marcos-delgado/GARD-model",
            "commit": subprocess.run(
                ["git", "-C", str(HISTORICAL_ROOT), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip(),
            "licenseStatus": "NO_LICENSE_DETECTED_REFERENCE_ONLY",
        },
        "references": [
            {
                "reference": 63,
                "identity": "Segre Ben-Eli Lancet 2001 J Theor Biol",
                "doi": "10.1006/jtbi.2001.2440",
                "url": "https://pubmed.ncbi.nlm.nih.gov/11735293/",
                "evidence": "public bibliographic abstract",
            },
            {
                "reference": 64,
                "identity": "Shenhav Oz Lancet 2001 BioSystems",
                "doi": "10.1023/A:1006583712886",
                "url": "https://pubmed.ncbi.nlm.nih.gov/11536890/",
                "evidence": "public bibliographic abstract and institutional PDF",
            },
            {
                "reference": 65,
                "identity": "Segre Ben-Eli Deamer Lancet 2000 PNAS",
                "doi": "10.1073/pnas.97.8.4112",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC18166/",
                "evidence": "open PMC full text",
            },
        ],
        "files": retained,
    }


def validate_l08_inputs() -> dict[str, Any]:
    manifest = pd.read_parquet(L08_ROOT / "trajectory_manifest.parquet")
    expected_groups = {
        ("A_FISSION_BOUNDARY", "CANDIDATE_2"),
        ("A_FISSION_BOUNDARY", "CANDIDATE_3"),
        ("B_HIGH_EXPOSURE", "CANDIDATE_2"),
        ("B_HIGH_EXPOSURE", "CANDIDATE_3"),
    }
    observed_groups = set(zip(manifest["mechanismId"], manifest["candidateId"], strict=False))
    if len(manifest) != 400 or observed_groups != expected_groups:
        raise RuntimeError("L08 trajectory manifest scope mismatch")
    checks: list[dict[str, Any]] = []
    for row in manifest.sort_values(["matrixIndex", "mechanismId", "candidateId"]).itertuples():
        path = Path(row.cachePath)
        observed = sha256_file(path)
        passed = bool(
            observed == row.cacheSha256
            and row.terminalStatus == "requested_fissions_completed"
            and int(row.completedFissions) == 100
        )
        checks.append(
            {
                "mechanismId": row.mechanismId,
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "trajectoryId": row.trajectoryId,
                "cachePath": str(path),
                "cacheSha256": observed,
                "manifestCacheSha256": row.cacheSha256,
                "trajectorySha256": row.trajectorySha256,
                "betaSha256": row.betaSha256,
                "initialStateSha256": row.initialStateSha256,
                "passed": passed,
            }
        )
    if not all(row["passed"] for row in checks):
        raise RuntimeError("L08 cache integrity failed")
    return {
        "schema": "eidosoma.e01.s19_l09.input_manifest.v1",
        "sourceLoop": "S19-L08",
        "matrixCount": 100,
        "trajectoryCount": 400,
        "primaryTrajectoryCount": 200,
        "comparatorTrajectoryCount": 200,
        "groups": sorted([list(item) for item in expected_groups]),
        "completeNoReplacement": True,
        "checks": checks,
    }


def benchmark_ten() -> dict[str, Any]:
    paths = [
        L08_CACHE / f"M{index:03d}__A_FISSION_BOUNDARY__CANDIDATE_2.pkl"
        for index in range(10)
    ]
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    completion = 0
    for path in paths:
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
        post = [item for item in selected if item.observation_kind == "post_fission"]
        boundary = np.asarray([item.state for item in post], dtype=np.float64)
        molecular = np.asarray([item.state for item in selected], dtype=np.float64)
        for pipeline_id, fit in (
            (R1_ID, fit_r1_historical(boundary, f"benchmark::{trajectory.trajectory_id}")),
            (R2_ID, fit_r2_euclidean(boundary, f"benchmark::{trajectory.trajectory_id}")),
        ):
            if fit.status == "ELIGIBLE":
                label_against_reference(molecular, fit.dominant_centroid)
            completion += int(pipeline_id in LABEL_IDS)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    projected_cpu_hours = cpu * (200 / 10) * 2.5 / 3600
    projected_wall_hours = wall * (200 / 10) * 2.5 / 8 / 3600
    return {
        "schema": "eidosoma.e01.s19_l09.preoutcome_benchmark.v1",
        "trajectoryCount": 10,
        "pipelineExecutions": completion,
        "scientificValuesRetainedOrOpened": False,
        "wallSeconds": wall,
        "cpuSeconds": cpu,
        "peakRssKiBBefore": peak_before,
        "peakRssKiBAfter": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "projectionSafetyFactor": 2.5,
        "projectedCpuHoursFullScope": projected_cpu_hours,
        "projectedWallHoursEightWorkers": projected_wall_hours,
        "cpuCeilingHours": 32,
        "wallCeilingHours": 8,
        "reserveFraction": 0.1,
        "passed": bool(projected_cpu_hours <= 28.8 and projected_wall_hours <= 7.2),
        "completedAtUtc": utc_now(),
    }


def append_preoutcome_root_ledgers(sources: dict[str, Any]) -> None:
    """Append L09 authorization, hypotheses, and source identities before outcomes."""

    now = utc_now()
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if "S19-L09" not in set(ledger["loopId"].astype(str)):
        row = {
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "timestampUtc": now,
            "loopId": "S19-L09",
            "recordPhase": "PRE_LOOP_OUTCOME_BLIND_METHOD_LOCK",
            "beliefBeforeLoop": "L08 ruled out its complete boundary-projection and high-exposure mechanisms, but neither directly implemented molecular membership relative to the run's dominant recurring composition.",
            "motivatingEvidence": "The paper explicitly describes composition-space attractor clusters and entry/exit relative to the most recurring composition; pinned GARD source defines non-drift filtering, compotype clustering, and most-frequent-compotype handling.",
            "failureOrAmbiguityTargeted": "Whether the 88%-versus-98% discrepancy is principally a self-replicator-label mismatch rather than exposure or denominator mismatch.",
            "selectedHypotheses": "Exactly R1 historical dominant compotype and R2 paper-Euclidean dominant attractor, each with direct molecular strict-H>0.9 membership.",
            "learned": "Pending locked L09 execution.",
            "weakenedHypotheses": "Pending locked L09 execution.",
            "remainingPlausibleHypotheses": "Both registered pipelines remain exploratory until validation; exact author code is unavailable.",
            "proposedNextTest": "Execute the pushed two-pipeline L09 contract once, then stop for mandatory human review.",
            "informationGainRationale": "The two pipelines separate direct historical-source lineage from the paper's Euclidean wording and test the label upstream of Figures 3-6 and Table 1 without using emergence to select it.",
            "appendOnly": True,
        }
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_parquet(ledger_path, index=False, compression="zstd")
        with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n## Entry 019 — S19-L09 pre-loop recurring-attractor lock\n\n"
                "- **Belief before:** L08 did not directly implement the paper's most-recurring-composition molecular label.\n"
                "- **Motivating evidence:** The paper names composition-space attractors and the most recurring composition; historical GARD provides non-drift, compotype, and most-frequent-compotype operations.\n"
                "- **Selected hypotheses:** Exactly R1 historical dominant compotype and R2 paper-Euclidean dominant attractor, both with direct strict-`H>0.9` molecular membership.\n"
                "- **What is not selectable:** H threshold, third clustering family, emergence association, prediction, intervention, or favorable simulator candidate.\n"
                "- **Expected information gain:** Distinguish label mismatch from L08's exposure/denominator mechanisms on the same frozen units.\n"
                "- **Next action:** Execute the pushed L09 lock once and stop for human review.\n"
            )

    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    additions = []
    if "S19-L09-R1" not in set(candidates["candidateId"].astype(str)):
        additions.extend(
            [
                {
                    "candidateId": "S19-L09-R1",
                    "bundleId": "L09_RECURRING_ATTRACTOR_LABEL_RECONSTRUCTION",
                    "selected": True,
                    "sourceGrounding": 5,
                    "paperFingerprintSpecificity": 5,
                    "explanatoryLeverage": 5,
                    "testability": 5,
                    "crossCandidateDiscriminability": 5,
                    "computeEfficiency": 5,
                    "independenceFromPriorOutcomeSelection": 3,
                    "outcomeGuidedThresholdSelection": 0,
                    "deterministicHReuse": 1,
                    "completedFitLeakage": 1,
                    "candidateSpecificSuccess": 0,
                    "undefinedAuthorSemantics": 2,
                    "branchCount": 1,
                    "proposedSpecification": "Pinned historical non-drift plus dominant cosine compotype; direct molecular H>0.9 membership",
                    "selectionReason": "Direct public GARD source lineage and explicit human authorization",
                    "rankingScore": 29.0,
                    "frozenRank": 1,
                    "registryOrder": int(candidates["registryOrder"].max()) + 1,
                },
                {
                    "candidateId": "S19-L09-R2",
                    "bundleId": "L09_RECURRING_ATTRACTOR_LABEL_RECONSTRUCTION",
                    "selected": True,
                    "sourceGrounding": 4,
                    "paperFingerprintSpecificity": 5,
                    "explanatoryLeverage": 5,
                    "testability": 5,
                    "crossCandidateDiscriminability": 5,
                    "computeEfficiency": 5,
                    "independenceFromPriorOutcomeSelection": 3,
                    "outcomeGuidedThresholdSelection": 0,
                    "deterministicHReuse": 1,
                    "completedFitLeakage": 1,
                    "candidateSpecificSuccess": 0,
                    "undefinedAuthorSemantics": 3,
                    "branchCount": 1,
                    "proposedSpecification": "All-boundary Euclidean k-means dominant attractor; direct molecular H>0.9 membership",
                    "selectionReason": "Paper-literal Euclidean attractor wording and explicit human authorization",
                    "rankingScore": 27.0,
                    "frozenRank": 2,
                    "registryOrder": int(candidates["registryOrder"].max()) + 2,
                },
            ]
        )
    if additions:
        candidates = pd.concat(
            [candidates, pd.DataFrame(additions, columns=candidates.columns)], ignore_index=True
        )
        candidates.to_parquet(candidate_path, index=False, compression="zstd")

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    source_ledger = pd.read_parquet(source_path)
    source_rows = []
    known = set(source_ledger["sourceId"].astype(str))
    source_map = {Path(item["path"]).name: item for item in sources["files"]}
    records = [
        ("L09_TARGET_PAPER_V1", "UPLOADED_ORIGINAL_PAPER", "local uploaded input", None, "arXiv:2607.28250v1", PAPER_MD, "INPUT_ATTACHMENT_CONTEXT", "Paper defines clusters, attractor-like homeostasis, Euclidean composition space, and most-recurring-composition entry/exit.", "INPUT_REFERENCE_ONLY"),
        ("L09_PINNED_GARD_COMPTYPE", "PUBLIC_CODE_LINEAGE", "https://github.com/marcos-delgado/GARD-model", "marcos-delgado/GARD-model", sources["historicalGard"]["commit"], HISTORICAL_ROOT / "tgs_acluster.m", "DIRECT_PUBLIC_CODE_LINEAGE", "Historical pipeline filters non-drift boundaries, clusters k=1-10 with ten replicas and early stop, and retains compotype centroids.", "IDENTITY_AND_FINDING_ONLY"),
        ("L09_PINNED_GARD_NONDRIFT", "PUBLIC_CODE_LINEAGE", "https://github.com/marcos-delgado/GARD-model", "marcos-delgado/GARD-model", sources["historicalGard"]["commit"], HISTORICAL_ROOT / "tgs_nondrift.m", "DIRECT_PUBLIC_CODE_LINEAGE", "Technique 1 averages incoming and outgoing adjacent-generation H and applies strict H>0.9.", "IDENTITY_AND_FINDING_ONLY"),
        ("L09_REFERENCE_65_PMC18166", "CITED_METHOD_FULL_TEXT", "https://pmc.ncbi.nlm.nih.gov/articles/PMC18166/", None, "DOI:10.1073/pnas.97.8.4112", Path("/cache/e01_s19_l09/sources/PMC18166.html"), "DIRECT_CITED_METHOD", "Defines H and homeostatic quasi-stationary compositional recurrence in the GARD lineage.", "IDENTITY_AND_FINDING_ONLY"),
        ("L09_REFERENCE_64", "CITED_METHOD", "https://pubmed.ncbi.nlm.nih.gov/11536890/", None, "DOI:10.1023/A:1006583712886", Path("/cache/e01_s19_l09/sources/ref64_GARD_domain.pdf"), "DIRECT_CITED_METHOD", "GARD-domain method context for compositional self-replication and recurrence.", "IDENTITY_AND_FINDING_ONLY"),
        ("L09_REFERENCE_63", "CITED_METHOD", "https://pubmed.ncbi.nlm.nih.gov/11735293/", None, "DOI:10.1006/jtbi.2001.2440", Path("/cache/e01_s19_l09/sources/PubMed_11735293.html"), "DIRECT_CITED_METHOD", "GARD heredity/composome method context.", "IDENTITY_AND_FINDING_ONLY"),
    ]
    for source_id, source_type, url, repo_identity, version, path, evidence, finding, redistribution in records:
        if source_id in known:
            continue
        source_rows.append(
            {
                "sourceId": source_id,
                "sourceType": source_type,
                "url": url,
                "repositoryIdentity": repo_identity,
                "commitOrVersion": version,
                "treeIdentity": None,
                "retrievalDate": "2026-08-09",
                "retainedPath": str(path),
                "sha256": sha256_file(path),
                "licenseStatus": "NO_LICENSE_DETECTED_REFERENCE_ONLY" if "GARD" in source_id else "PUBLIC_INPUT_OR_CITED_SOURCE_REFERENCE_ONLY",
                "evidenceClass": evidence,
                "finding": finding,
                "redistributionStatus": redistribution,
            }
        )
    if source_rows:
        source_ledger = pd.concat(
            [source_ledger, pd.DataFrame(source_rows, columns=source_ledger.columns)],
            ignore_index=True,
        )
        source_ledger.to_parquet(source_path, index=False, compression="zstd")
        with (ARTIFACT_ROOT / "source_search_report.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n## S19-L09 additive source audit — dominant recurring composition\n\n"
                "The target paper directly describes self-replicators as composition-space clusters with attractor-like homeostasis and entry/exit relative to the most recurring composition. The pinned historical GARD lineage independently fixes strict cosine `H>0.9`, adjacent-generation non-drift smoothing, k=1–10 compotype clustering, ten replicas, silhouette selection, a four-k nonimprovement stop, and most-frequent-compotype handling. References 63–65 ground H, homeostatic quasi-stationary composomes, and the GARD inheritance lineage. The paper does not identify the historical MATLAB version/RNG or its exact target implementation; R1 and R2 remain explicit reconstructions, no author was contacted, and unlicensed source was not redistributed.\n"
            )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not any(item["loopId"] == "S19-L09" for item in registry["loops"]):
        registry["loops"].append(
            {
                "loopId": "S19-L09",
                "versionedLoopId": VERSION,
                "status": "AUTHORIZED_PREOUTCOME_LOCK_PREPARED",
                "authorized": True,
                "outcomeAccessed": False,
                "humanReviewRequiredAfter": True,
                "completed": False,
                "eligibleScientificResults": None,
                "promotedLeadCount": 0,
                "nextStepActive": True,
            }
        )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not any(item.get("scope") == VERSION for item in review["history"]):
        review["history"].append(
            {
                "date": "2026-08-09",
                "decision": "AUTHORIZE_S19_L09_RECURRING_ATTRACTOR_LABEL_RECONSTRUCTION_ONLY",
                "scope": VERSION,
                "source": "explicit_human_direction",
            }
        )
    review["pendingDecision"] = "S19_L09_ACTIVE_MANDATORY_HUMAN_REVIEW_AFTER_COMPLETION"
    write_json(review_path, review)


def main() -> None:
    started = utc_now()
    preexisting = LOOP_ROOT.exists()
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if preexisting and not (LOOP_ROOT / "prepare_runtime.json").exists():
        write_json(
            LOOP_ROOT / "preoutcome_preparation_failure_001.json",
            {
                "schema": "eidosoma.e01.s19_l09.preoutcome_preparation_failure.v1",
                "failureId": "S19-L09-PRE-F001",
                "phase": "TEN_TRAJECTORY_OPAQUE_BENCHMARK",
                "outcomesRetainedOrOpened": False,
                "error": "R2 arithmetic centroid contained negative machine-epsilon residue and the initial nonnegative reference validator rejected it",
                "resolution": "Before commit/push and before scientific outcome access, clamp only centroid coordinates >=-1e-12 to zero; any smaller negative fails closed",
                "scientificSettingsChanged": False,
                "thresholdChanged": False,
                "pipelineFamilyChanged": False,
                "valuePreservingAmendment": True,
                "recordedAtUtc": utc_now(),
            },
        )
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["versionedLoopId"] != VERSION:
        raise RuntimeError("config version mismatch")

    baseline = immutable_rows()
    write_json(
        LOOP_ROOT / "immutable_prior_baseline.json",
        {
            "schema": "eidosoma.e01.s19_l09.immutable_prior_baseline.v1",
            "capturedAtUtc": utc_now(),
            "fileCount": len(baseline),
            "totalBytes": sum(row["bytes"] for row in baseline),
            "files": baseline,
        },
    )
    inputs = validate_l08_inputs()
    write_json(LOOP_ROOT / "input_manifest.json", inputs)
    sources = source_manifest()
    write_json(LOOP_ROOT / "source_snapshot_manifest.json", sources)

    fixtures = run_fixtures()
    fixtures.to_parquet(
        LOOP_ROOT / "synthetic_fixture_results.parquet", index=False, compression="zstd"
    )
    fixtures.loc[fixtures["fixtureId"] == "F01_TGS_NONDRIFT_TECHNIQUE1"].to_csv(
        LOOP_ROOT / "source_equivalence_results.csv", index=False, lineterminator="\n"
    )
    write_json(
        LOOP_ROOT / "fixture_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l09.fixture_manifest.v1",
            "fixtureCount": int(fixtures["fixtureId"].nunique()),
            "checkCount": int(len(fixtures)),
            "passedCount": int(fixtures["passed"].sum()),
            "failedCount": int((~fixtures["passed"]).sum()),
            "allMandatoryPassed": bool(fixtures["passed"].all()),
            "resultsPath": str(LOOP_ROOT / "synthetic_fixture_results.parquet"),
            "resultsSha256": sha256_file(LOOP_ROOT / "synthetic_fixture_results.parquet"),
        },
    )

    table_lock = {
        "schema": "eidosoma.e01.s19_l09.table1_semantics_lock.v1",
        "probability": "P=sum(Y_t)/T",
        "persistence": "L=sum(Y_t)",
        "consistency": "PearsonCorr(Y_t,Y_{t+1}); undefined for constant or too-short sequence",
        "firstOnset": {
            "zeroBasedRawMolecularStep": "min{t:Y_t=1}",
            "oneBasedRawMolecularStep": "min{t:Y_t=1}+1",
            "normalizedFraction": "t/max(1,T-1)",
            "fissionGeneration": "growth_generation_one_based at onset",
        },
        "dispersion": ["sample standard deviation", "standard error"],
        "authorDispersionIdentity": "AUTHOR_DISPERSION_UNRESOLVED",
        "boundaryAndMolecularNoninterchangeable": True,
    }
    write_yaml(LOOP_ROOT / "table1_semantics_lock.yaml", table_lock)
    write_yaml(
        LOOP_ROOT / "label_pipeline_registry.yaml",
        {
            "schema": "eidosoma.e01.s19_l09.label_pipeline_registry.v1",
            "pipelineCount": 2,
            "pipelines": config["labels"]["pipelines"],
            "commonMembership": {
                "similarity": "pinned historical tgs_H cosine",
                "threshold": THRESHOLD,
                "comparator": "strict greater than",
                "primaryObject": "selected molecular composition directly to dominant centroid",
                "boundaryObject": "diagnostic only",
            },
        },
    )

    code_paths = [CORE_PATH, PREPARE_PATH, RUNNER_PATH, TEST_PATH, CONFIG_PATH]
    for path in code_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    method_lock = {
        "schema": "eidosoma.e01.s19_l09.label_method_lock.v1",
        "versionedLoopId": VERSION,
        "lockedAtUtc": utc_now(),
        "outcomesOpened": False,
        "configuration": config,
        "code": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for path in code_paths
        ],
        "sourceSnapshotManifestSha256": sha256_file(
            LOOP_ROOT / "source_snapshot_manifest.json"
        ),
        "inputManifestSha256": sha256_file(LOOP_ROOT / "input_manifest.json"),
        "fixtureManifestSha256": sha256_file(LOOP_ROOT / "fixture_manifest.json"),
        "threshold": THRESHOLD,
        "pipelineIds": list(LABEL_IDS),
        "kValues": list(K_VALUES),
        "replicas": REPLICAS,
        "earlyStopStreakR1": EARLY_STOP_STREAK,
        "maximumIterations": MAX_ITERATIONS,
        "minimumValidClusterSize": MINIMUM_VALID_CLUSTER_SIZE,
        "minimumStrictH090Visits": MINIMUM_REFERENCE_MEMBERSHIP_VISITS,
        "randomReferenceDraws": RANDOM_REFERENCE_DRAWS,
        "bootstrapReplicates": BOOTSTRAP_REPLICATES,
        "rootSeedHexSha256": hashlib.sha256(ROOT_SEED_HEX.encode()).hexdigest(),
        "scientificAmendmentsPermittedAfterRelease": False,
    }
    write_json(LOOP_ROOT / "label_method_lock.json", method_lock)

    write_yaml(
        LOOP_ROOT / "preregistration.yaml",
        {
            "researchStepId": "S19-L09",
            "versionedLoopId": VERSION,
            "status": "PREOUTCOME_LOCKED_PENDING_PUSH",
            "centralQuestion": "Can one of exactly two source/paper-grounded dominant recurring composition labels improve the complete paper control fingerprint in both candidates?",
            "inputs": config["inputs"],
            "labels": config["labels"],
            "table1": config["table1"],
            "controls": config["controls"],
            "statistics": config["statistics"],
            "successAndPromotion": {
                "definedBothCandidatesMinimum": 95,
                "occupancyRange": [0.85, 0.91],
                "persistenceRange": [518, 914],
                "consistencyCloserThanEveryComparator": True,
                "onsetCloserThanEveryComparator": True,
                "nontrivialPreOnset": config["promotionOperationalization"]["nontrivialPreOnset"],
                "meaningfulQuarterEligible": config["promotionOperationalization"]["meaningfulQuarterEligibility"],
                "nondegeneratePositiveNegativeEpisodes": config["promotionOperationalization"]["nondegenerateEpisodes"],
                "largestBeatsRandomAndSecond": True,
                "bothCandidatesAgree": True,
                "exactReplayAndValidation": True,
                "noEmergenceSelection": True,
                "maximumPromotedLeads": 1,
            },
            "classificationOrder": [
                "PROMOTABLE_TO_UNTOUCHED_CONFIRMATION",
                "EXPLORATORY_PAPER_MATCH_OCCUPANCY_ONLY",
                "METHOD_DEPENDENT_LEAD",
                "RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED",
                "LOOP_FAILED_CLOSED",
            ],
            "compute": config["compute"],
            "stopConditions": config["stopConditions"],
        },
    )

    (LOOP_ROOT / "decision_record.md").write_text(
        """# S19-L09 Decision Record\n\n"
        "- **Research step:** `E01-S19-L09-RECURRING-ATTRACTOR-LABEL-RECONSTRUCTION-v1.0.0`\n"
        "- **Status:** authorized; outcome-blind method lock prepared.\n"
        "- **Artifacts written:** preregistration, source/paper audit, source snapshot, Table 1 lock, label-method lock, fixture evidence, label registry, input/immutable baselines, and benchmark.\n"
        "- **Validation result:** mandatory source/synthetic fixtures passed before scientific execution; repository push and release gate remain required.\n"
        "- **Outcome classification:** pending; no L09 label outcome was retained or opened.\n"
        "- **Caveats/blockers:** historical MATLAB RNG/version behavior and the paper's exact clustering code are unavailable; those choices are explicitly reconstructed and frozen.\n"
        "- **Recommended next action:** commit and push the complete lock, verify a clean worktree, execute L09 once, validate, and stop for human review.\n\n"
        "This additive decision preserves all S01–S18 and S19-L01–L08 evidence. It permits exactly two label pipelines and no new trajectory, emergence, prediction, or intervention computation.\n",
        encoding="utf-8",
    )

    (LOOP_ROOT / "source_and_paper_label_audit.md").write_text(
        """# Source and Paper Label Audit\n\n"
        "## Top summary\n\n"
        "- **Research step:** S19-L09.\n"
        "- **Status:** pre-outcome source audit complete.\n"
        "- **Artifacts written:** this audit, the hashed source snapshot, Table 1 lock, label-method lock, and fixture evidence.\n"
        "- **Validation result:** pinned historical operations and mandatory fixtures passed; every retained source has a SHA-256 identity.\n"
        "- **Outcome classification:** no scientific label outcome accessed.\n"
        "- **Caveats/blockers:** no authoritative paper code; historical MATLAB release/RNG details are unavailable; paper Table 1 dispersion and onset units remain unresolved.\n"
        "- **Recommended next action:** execute only the pushed two-pipeline lock.\n\n"
        "## Direct paper evidence\n\n"
        "The paper describes recurring compositions inherited across generations, calls the self-replicators clusters in molecular-composition space with homeostatic attractor-like growth, and says entry/exit depends on similarity to the run's most recurring composition. Its Methods separately describes highly similar steady compositions in Euclidean space. Figure 1 and Table 1 were therefore treated as measurement semantics, not as permission to tune a cluster radius or H threshold.\n\n"
        "## Direct historical-source evidence\n\n"
        "The pinned GARD v10 lineage defines H as clipped cosine similarity; technique 1 marks a boundary non-drift when the average of incoming and outgoing adjacent-generation H exceeds 0.9, duplicating the first/last adjacent score at the endpoints. `tgs_acluster` clusters only non-drift boundaries, evaluates k=1–10 with ten replicas, chooses replicas by minimum distance, scores k>1 by mean silhouette, uses a special mean-H carpet score for k=1, and stops after four k values without improvement. `getcomposometime_v10` and `biased_gard_v10` identify the most frequent compotype.\n\n"
        "## Reconstruction choices\n\n"
        "R1 reproduces those operations with deterministic CPU-float64 spherical k-means because the original MATLAB release and RNG behavior are not identified. R2 follows the paper's Euclidean wording with deterministic Lloyd k-means. R2's k=1 silhouette is explicitly undefined and cannot win silhouette selection; both pipelines require at least two assigned members and two strict-H>0.9 centroid visits for a recurring reference. These are frozen reconstruction choices, not author-code claims.\n\n"
        "## Cited-method context\n\n"
        "References 63–65 ground the GARD/composome lineage. The open PNAS text defines H and homeostatic quasi-stationary composomes; the related GARD papers describe compotypes/compositional recurrence. Public identities, retrieval paths, hashes, and licensing/redistribution status are in the source snapshot.\n\n"
        "## Table 1 semantics\n\n"
        "Molecular probability, persistence, consecutive-label Pearson consistency, and onset are frozen exactly as specified in `table1_semantics_lock.yaml`. Zero-based, one-based, normalized, and fission-generation onset are all reported. Both SD and SE are reported; `AUTHOR_DISPERSION_UNRESOLVED` cannot be resolved by target proximity. Boundary diagnostics cannot replace molecular results.\n",
        encoding="utf-8",
    )

    append_preoutcome_root_ledgers(sources)

    benchmark = benchmark_ten()
    write_json(LOOP_ROOT / "preoutcome_benchmark.json", benchmark)
    if not benchmark["passed"]:
        raise RuntimeError("pre-outcome benchmark exceeds reserved compute ceiling")

    write_json(
        LOOP_ROOT / "prepare_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l09.prepare_runtime.v1",
            "startedAtUtc": started,
            "completedAtUtc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikitLearn": sklearn.__version__,
            "pandas": pd.__version__,
            "workerThreads": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "repository": {
                "branch": git("branch", "--show-current"),
                "headBeforeLockCommit": git("rev-parse", "HEAD"),
                "worktreeDirtyExpectedBeforeLockCommit": bool(git("status", "--porcelain=v1")),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "PREOUTCOME_LOCK_PREPARED",
                "fixtureChecks": len(fixtures),
                "immutableFiles": len(baseline),
                "inputTrajectories": len(inputs["checks"]),
                "projectedCpuHours": benchmark["projectedCpuHoursFullScope"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
