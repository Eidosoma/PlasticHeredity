#!/usr/bin/env python3
"""Materialize the pushed, outcome-blind E01/S19-L06R repair lock."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.e01.prepare_s19_l06_lock as l06prep
from e01_s19_boundary_recurrence.core import (
    COMPARATOR_LABEL_ID,
    STRUCTURAL_LABEL_ID,
    boundary_recurrence,
    boundary_recurrence_reference,
)
from e01_s19_boundary_recurrence_repair.core import (
    ABSOLUTE_TOLERANCE,
    LOOP_ID,
    MAXIMUM_ULP_DISTANCE,
    RELATIVE_TOLERANCE,
    VERSION,
    compare_discrete_recurrence,
    compare_float64_scores,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L06R"
L06_ROOT = ARTIFACT_ROOT / "loops/L06"
CACHE_ROOT = Path("/cache/e01_s19_l06r")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
PREREG = REPO_ROOT / "configs/e01/s19_l06r_preregistration.yaml"
METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l06r_method_lock.json"
L06_METHOD_LOCK = REPO_ROOT / "configs/e01/s19_l06_method_lock.json"
S06_PRECISION = REPO_ROOT / "configs/e01/s06_precision_contract.yaml"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def repository_lock() -> dict[str, object]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    clean = not bool(git("status", "--porcelain=v1"))
    passed = branch == "eidosoma/groups/42" and head == remote and clean
    result = {
        "schema": "eidosoma.e01.s19_l06r_preoutcome_repository_lock.v1",
        "loopId": LOOP_ID,
        "branch": branch,
        "head": head,
        "remoteHead": remote,
        "cleanWorktree": clean,
        "headEqualsRemote": head == remote,
        "outcomeAccessed": False,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"clean pushed repair lock failed: {result}")
    return result


def manifest_rows(root: Path, role: str) -> list[dict[str, object]]:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("passed"):
        raise RuntimeError(f"prior manifest is not passing: {manifest_path}")
    rows: list[dict[str, object]] = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        actual = sha256_file(path) if path.is_file() else None
        size = path.stat().st_size if path.is_file() else None
        if actual != entry["sha256"] or size != entry["bytes"]:
            raise RuntimeError(f"changed immutable artifact: {path}")
        rows.append({"path": str(path), "role": role, "bytes": size, "sha256": actual})
    rows.append(
        {
            "path": str(manifest_path),
            "role": role,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
    )
    return rows


def immutable_baseline() -> tuple[dict[str, object], dict[str, object]]:
    s18 = json.loads(
        (ARTIFACT_ROOT / "s18_immutable_baseline.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, object]] = []
    for entry in s18["files"]:
        path = Path(entry["path"])
        actual = sha256_file(path) if path.is_file() else None
        size = path.stat().st_size if path.is_file() else None
        if actual != entry["sha256"] or size != entry["bytes"]:
            raise RuntimeError(f"changed S18 historical artifact: {path}")
        rows.append(
            {"path": str(path), "role": entry["role"], "bytes": size, "sha256": actual}
        )
    for loop in ("L01", "L02", "L03", "L04", "L05", "L06"):
        rows.extend(
            manifest_rows(ARTIFACT_ROOT / f"loops/{loop}", f"IMMUTABLE_S19_{loop}")
        )
    by_path = {str(row["path"]): row for row in rows}
    ordered = [by_path[key] for key in sorted(by_path)]
    digest = hashlib.sha256()
    for row in ordered:
        digest.update(canonical_json(row).encode())
        digest.update(b"\n")
    payload = {
        "schema": "eidosoma.e01.s19_l06r_immutable_prior_baseline.v1",
        "loopId": LOOP_ID,
        "historicalBoundary": "S01-S18_V1_V2_S19-L01_THROUGH_L06_AND_S17_WAIVER",
        "fileCount": len(ordered),
        "totalBytes": int(sum(int(row["bytes"]) for row in ordered)),
        "aggregateSha256": digest.hexdigest(),
        "files": ordered,
    }
    validation = {
        "schema": "eidosoma.e01.s19_l06r_immutable_prior_validation.v1",
        "fileCount": len(ordered),
        "aggregateSha256": digest.hexdigest(),
        "missing": [],
        "mismatches": [],
        "passed": True,
    }
    return payload, validation


def synthetic_validation_and_benchmark() -> tuple[
    pd.DataFrame, dict[str, object], dict[str, object]
]:
    trajectory = l06prep.synthetic_trajectory()
    selected = l06prep.selected_clock_observations(
        trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
    )
    states = np.asarray([item.state for item in selected], dtype=np.int64)
    generations = np.asarray(
        [item.growth_generation_one_based for item in selected], dtype=np.int64
    )
    kinds = np.asarray([item.observation_kind for item in selected], dtype=str)
    indices = np.arange(len(selected), dtype=np.int64)
    started = time.process_time()
    canonical = boundary_recurrence(states, generations, kinds, indices)
    independent = boundary_recurrence_reference(states, generations, kinds, indices)
    elapsed = time.process_time() - started
    score = compare_float64_scores(canonical["scores"], independent["scores"])
    discrete = compare_discrete_recurrence(canonical, independent)
    fixtures = [
        {
            "fixtureId": "SYNTHETIC_L06_BOUNDARY_TRAJECTORY",
            "expected": "PASS",
            **score,
            "discreteExact": bool(all(discrete.values())),
        },
    ]
    base = np.asarray([0.91], dtype=np.float64)
    eight = base.copy()
    nine = base.copy()
    for _ in range(8):
        eight = np.nextafter(eight, np.inf)
    for _ in range(9):
        nine = np.nextafter(nine, np.inf)
    for fixture_id, right, expected in (
        ("EIGHT_ULP_BOUNDARY", eight, "PASS"),
        ("NINE_ULP_BOUNDARY", nine, "FAIL"),
    ):
        result = compare_float64_scores(base, right)
        fixtures.append(
            {
                "fixtureId": fixture_id,
                "expected": expected,
                **result,
                "discreteExact": True,
            }
        )
    frame = pd.DataFrame(fixtures)
    passed = bool(
        frame.loc[
            frame["fixtureId"].eq("SYNTHETIC_L06_BOUNDARY_TRAJECTORY"), "passed"
        ].all()
        and frame.loc[frame["fixtureId"].eq("EIGHT_ULP_BOUNDARY"), "passed"].all()
        and not frame.loc[frame["fixtureId"].eq("NINE_ULP_BOUNDARY"), "passed"].any()
        and frame.loc[
            frame["fixtureId"].eq("SYNTHETIC_L06_BOUNDARY_TRAJECTORY"), "discreteExact"
        ].all()
    )
    validation = {
        "schema": "eidosoma.e01.s19_l06r_synthetic_fixture_validation.v1",
        "fixtureCount": len(frame),
        "passed": passed,
    }
    l06_benchmark = l06prep.synthetic_benchmark()
    projection = float(
        l06_benchmark["projectedScientificCpuHours"] + elapsed * 200 * 3 / 3600
    )
    total = projection + 3.2
    benchmark = {
        "schema": "eidosoma.e01.s19_l06r_compute_benchmark.v1",
        "syntheticClockRows": len(selected),
        "numericalReplayCpuSeconds": elapsed,
        "inheritedL06ProjectedScientificCpuHours": l06_benchmark[
            "projectedScientificCpuHours"
        ],
        "projectedScientificCpuHours": projection,
        "reservedValidationFinalizationCpuHours": 3.2,
        "projectedTotalCpuHours": total,
        "cpuCeilingHours": 32.0,
        "wallCeilingHours": 8.0,
        "gpuCeilingHours": 0.0,
        "gatePassed": bool(total <= 32.0),
    }
    return frame, validation, benchmark


def append_root_ledgers(timestamp: str) -> None:
    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    ids = ["S19-L06R-LABEL-01", "S19-L06R-LABEL-02"]
    if candidates["candidateId"].isin(ids).any():
        raise RuntimeError("L06R candidate registry rows already exist")
    prior = candidates.loc[
        candidates["candidateId"].isin(["S19-L06-LABEL-01", "S19-L06-LABEL-02"])
    ].copy()
    prior["candidateId"] = ids
    prior["bundleId"] = "L06R_NUMERICAL_EQUIVALENCE_REPAIR_UNCHANGED_L06_SCIENCE"
    prior["selectionReason"] = (
        "human-directed one-repair numerical-equivalence rerun; scientific labels unchanged"
    )
    prior["registryOrder"] = np.arange(len(candidates) + 1, len(candidates) + 3)
    pd.concat([candidates, prior[candidates.columns]], ignore_index=True).to_parquet(
        candidate_path, index=False
    )

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    additions = pd.DataFrame(
        [
            {
                "sourceId": "L06R_S06_NUMERICAL_EQUIVALENCE_POLICY",
                "sourceType": "FROZEN_INTERNAL_METHOD_CONTRACT",
                "url": None,
                "repositoryIdentity": "Eidosoma/arrival-of-self-replicators",
                "commitOrVersion": "E01-trajectory-precision-v1.0.0",
                "treeIdentity": None,
                "retrievalDate": timestamp[:10],
                "retainedPath": str(S06_PRECISION),
                "sha256": sha256_file(S06_PRECISION),
                "licenseStatus": "INTERNAL_METHOD_CONTRACT",
                "evidenceClass": "DIRECT_FROZEN_NUMERICAL_POLICY",
                "finding": "S06 freezes all-three abs<=1e-12, rel<=1e-12, and ULP<=8 comparison for corresponding finite values while discrete fields remain exact.",
                "redistributionStatus": "INTERNAL_OR_CITABLE_REFERENCE",
            },
            {
                "sourceId": "L06R_FAILED_L06_IMMUTABLE_CONTEXT",
                "sourceType": "FROZEN_INTERNAL_EVIDENCE",
                "url": None,
                "repositoryIdentity": None,
                "commitOrVersion": "S19-L06_FAILED_CLOSED",
                "treeIdentity": None,
                "retrievalDate": timestamp[:10],
                "retainedPath": str(L06_ROOT / "artifact_manifest.json"),
                "sha256": sha256_file(L06_ROOT / "artifact_manifest.json"),
                "licenseStatus": "INTERNAL_GENERATED_EVIDENCE",
                "evidenceClass": "FROZEN_INTERNAL_EVIDENCE",
                "finding": "L06 remains failed closed and supplies only the diagnosed numerical-replay discrepancy and unchanged scientific lock.",
                "redistributionStatus": "INTERNAL_OR_CITABLE_REFERENCE",
            },
        ]
    )[sources.columns]
    if sources["sourceId"].isin(additions["sourceId"]).any():
        raise RuntimeError("L06R source rows already exist")
    pd.concat([sources, additions], ignore_index=True).to_parquet(
        source_path, index=False
    )

    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if ledger["loopId"].eq(LOOP_ID).any():
        raise RuntimeError("L06R ledger row already exists")
    row = {
        "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
        "timestampUtc": timestamp,
        "loopId": LOOP_ID,
        "recordPhase": "PRE_LOOP_ADDITIVE_REPAIR_BELIEF_AND_SELECTION",
        "beliefBeforeLoop": "L06's bit-exact score gate may have rejected mathematically equivalent CPU-float64 reduction orders despite exact labels and recurrence counts.",
        "motivatingEvidence": "The diagnosed L06 maximum absolute score discrepancy was 3.3306690738754696e-16 while every diagnosed discrete label and recurrence count agreed.",
        "failureOrAmbiguityTargeted": "Whether every canonical/independent score pair across all 200 frozen trajectories satisfies the already documented S06 numerical-equivalence contract.",
        "selectedHypotheses": "One additive repair only: exact discrete replay plus abs<=1e-12, rel<=1e-12, and ULP<=8 for every finite score pair; all L06 science unchanged.",
        "learned": None,
        "weakenedHypotheses": None,
        "remainingPlausibleHypotheses": None,
        "proposedNextTest": "Complete L06R once if the numerical gate passes, then mandatory human review.",
        "informationGainRationale": "The repair directly adjudicates a diagnosed floating-point-order dependency without adding a label, threshold, dataset, seed, statistic, or promotion branch.",
        "appendOnly": True,
    }
    pd.concat(
        [ledger, pd.DataFrame([row])[ledger.columns]], ignore_index=True
    ).to_parquet(ledger_path, index=False)
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            "\n## Entry 013 — S19-L06R pre-loop additive repair lock\n\n"
            "- **Belief before the repair:** L06 may have rejected numerically equivalent float64 reduction orders while preserving every discrete outcome.\n"
            "- **Motivating evidence:** The diagnosed maximum absolute discrepancy was `3.3306690738754696e-16`; diagnosed labels and recurrence counts were exact.\n"
            "- **Repair selected:** One all-three-bound numerical policy (`abs<=1e-12`, `rel<=1e-12`, `ULP<=8`) with exact masks, labels, counts, and identities.\n"
            "- **Scientific changes:** None; L06 remains immutable and failed closed.\n"
            "- **Next action:** Execute L06R once from fresh cache, then stop for human review.\n"
        )
    with (ARTIFACT_ROOT / "source_search_report.md").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            "\n## S19-L06R additive numerical-policy refresh\n\n"
            "L06R adds no scientific source search. It uses the frozen S06 numerical-equivalence policy and the immutable failed-L06 record. The original paper, public source context, S13Y inputs, and L06 scientific formula remain unchanged.\n"
        )
    with (ARTIFACT_ROOT / "continuation_decision.md").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            "\n## Additive human decision — authorize S19-L06R only\n\n"
            "The human authorized `E01-S19-L06R-NUMERICAL-EQUIVALENCE-CONFIRMATION-v1.0.0` as a one-repair-only additive rerun. Failed L06 remains immutable. No L07, S20, E02, author contact, or report generation is active.\n"
        )
    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if any(item["loopId"] == LOOP_ID for item in registry["loops"]):
        raise RuntimeError("L06R registry row already exists")
    registry["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "PREANALYSIS_REPLAY_PASSED_READY_FOR_LOCKED_EXECUTION",
            "authorized": True,
            "outcomeAccessed": False,
            "humanReviewRequiredAfter": True,
            "completed": False,
            "eligibleScientificResults": False,
            "additiveRepairOf": "S19-L06",
            "repairCount": 1,
        }
    )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["history"].append(
        {
            "date": timestamp[:10],
            "decision": "AUTHORIZE_ADDITIVE_S19_L06R_OPTION_1_ONLY",
            "scope": VERSION,
            "source": "explicit_human_direction",
        }
    )
    history["pendingDecision"] = "POST_S19_L06R_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def main() -> None:
    started = datetime.now(timezone.utc)
    if LOOP_ROOT.exists():
        raise RuntimeError("L06R artifact directory already exists; overwrite refused")
    LOOP_ROOT.mkdir(parents=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    repository = repository_lock()
    baseline, immutable = immutable_baseline()
    fixtures, fixture_validation, benchmark = synthetic_validation_and_benchmark()
    replay_rows, replay = l06prep.exact_preanalysis_replay()
    if not (
        repository["passed"]
        and immutable["passed"]
        and fixture_validation["passed"]
        and benchmark["gatePassed"]
        and replay["passed"]
    ):
        raise RuntimeError("L06R pre-outcome gate failed")

    shutil.copy2(PREREG, LOOP_ROOT / "preregistration.yaml")
    shutil.copy2(METHOD_LOCK, LOOP_ROOT / "method_lock.json")
    for name in ("seed_manifest.parquet", "untouched_s20_design.yaml"):
        shutil.copy2(L06_ROOT / name, LOOP_ROOT / name)
    # New registries identify the additive wrapper while preserving the two L06 specifications.
    l06_candidates = pd.read_csv(L06_ROOT / "candidate_ranking.csv")
    l06_candidates["candidateId"] = ["S19-L06R-LABEL-01", "S19-L06R-LABEL-02"]
    l06_candidates["bundleId"] = (
        "L06R_NUMERICAL_EQUIVALENCE_REPAIR_UNCHANGED_L06_SCIENCE"
    )
    l06_candidates.to_csv(LOOP_ROOT / "candidate_ranking.csv", index=False)
    (LOOP_ROOT / "candidate_bundle_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "eidosoma.e01.s19_l06r_candidate_bundle_registry.v1",
                "loopId": LOOP_ID,
                "bundleCount": 1,
                "bundles": [
                    {
                        "bundleId": "L06R_NUMERICAL_EQUIVALENCE_REPAIR_UNCHANGED_L06_SCIENCE",
                        "repairOnly": True,
                        "inheritedScientificLoop": "S19-L06",
                        "labelIds": [COMPARATOR_LABEL_ID, STRUCTURAL_LABEL_ID],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    labels = yaml.safe_load(
        (L06_ROOT / "label_registry.yaml").read_text(encoding="utf-8")
    )
    labels["schema"] = "eidosoma.e01.s19_l06r_label_registry.v1"
    labels["loopId"] = LOOP_ID
    labels["inheritedScientificLoop"] = "S19-L06"
    (LOOP_ROOT / "label_registry.yaml").write_text(
        yaml.safe_dump(labels, sort_keys=False), encoding="utf-8"
    )
    pd.DataFrame(labels["labels"]).to_parquet(
        LOOP_ROOT / "label_registry.parquet", index=False
    )
    specs = pd.read_parquet(L06_ROOT / "specification_ledger.parquet").copy()
    specs["loopId"] = LOOP_ID
    specs["specificationId"] = specs["specificationId"].str.replace(
        "L06-", "L06R-", regex=False
    )
    specs.to_parquet(LOOP_ROOT / "specification_ledger.parquet", index=False)

    write_json(LOOP_ROOT / "preoutcome_repository_lock.json", repository)
    write_json(LOOP_ROOT / "immutable_prior_baseline.json", baseline)
    write_json(LOOP_ROOT / "immutable_prior_validation.json", immutable)
    write_json(LOOP_ROOT / "compute_benchmark.json", benchmark)
    fixtures.to_parquet(LOOP_ROOT / "synthetic_fixture_results.parquet", index=False)
    write_json(LOOP_ROOT / "synthetic_fixture_validation.json", fixture_validation)
    replay_rows.to_parquet(
        LOOP_ROOT / "preanalysis_replay_evidence.parquet", index=False
    )
    write_json(LOOP_ROOT / "preanalysis_replay_validation.json", replay)
    write_json(
        LOOP_ROOT / "numerical_equivalence_contract.json",
        {
            "schema": "eidosoma.e01.s19_l06r_numerical_equivalence_contract.v1",
            "loopId": LOOP_ID,
            "canonicalCpuFloat64Path": "boundary_recurrence",
            "independentCpuFloat64Path": "boundary_recurrence_reference",
            "finiteNonfiniteMasksExact": True,
            "nonfiniteClassesExact": True,
            "booleanLabelsExact": True,
            "recurrenceCountsAndIdentitiesExact": True,
            "absoluteTolerance": ABSOLUTE_TOLERANCE,
            "relativeTolerance": RELATIVE_TOLERANCE,
            "maximumUlpDistance": MAXIMUM_ULP_DISTANCE,
            "finitePassRule": "ALL_THREE_BOUNDS_EVERY_FINITE_PAIR",
            "scientificChanges": [],
            "secondRepairPermitted": False,
        },
    )
    write_json(
        LOOP_ROOT / "scientific_contract_reuse_validation.json",
        {
            "schema": "eidosoma.e01.s19_l06r_scientific_contract_reuse_validation.v1",
            "l06MethodLockPath": str(L06_METHOD_LOCK),
            "l06MethodLockSha256": sha256_file(L06_METHOD_LOCK),
            "seedManifestByteExact": sha256_file(LOOP_ROOT / "seed_manifest.parquet")
            == sha256_file(L06_ROOT / "seed_manifest.parquet"),
            "untouchedS20DesignByteExact": sha256_file(
                LOOP_ROOT / "untouched_s20_design.yaml"
            )
            == sha256_file(L06_ROOT / "untouched_s20_design.yaml"),
            "thresholdChanged": False,
            "recurrenceRuleChanged": False,
            "statisticsChanged": False,
            "promotionGatesChanged": False,
            "passed": True,
        },
    )
    write_json(
        LOOP_ROOT / "seed_reuse_validation.json",
        {
            "schema": "eidosoma.e01.s19_l06r_seed_reuse_validation.v1",
            "policy": "EXACT_L06_SCIENTIFIC_SEEDS_REUSED_NO_NEW_STREAM",
            "byteExact": sha256_file(LOOP_ROOT / "seed_manifest.parquet")
            == sha256_file(L06_ROOT / "seed_manifest.parquet"),
            "passed": True,
        },
    )
    inputs = [
        S13Y_ROOT / "trajectory_manifest.parquet",
        S13Y_ROOT / "label_values.parquet",
        L06_ROOT / "artifact_manifest.json",
        L06_ROOT / "classification.json",
        L06_METHOD_LOCK,
        S06_PRECISION,
    ]
    write_json(
        LOOP_ROOT / "input_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l06r_input_manifest.v1",
            "loopId": LOOP_ID,
            "trajectoryCount": 200,
            "sharedMatrixCount": 100,
            "l06WorkerCacheUse": False,
            "freshCacheRoot": str(CACHE_ROOT),
            "inputs": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in inputs
            ],
        },
    )
    write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l06r_source_snapshot_manifest.v1",
            "sources": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (
                    PREREG,
                    METHOD_LOCK,
                    L06_METHOD_LOCK,
                    S06_PRECISION,
                    REPO_ROOT / "src/e01_s19_boundary_recurrence/core.py",
                    REPO_ROOT / "src/e01_s19_boundary_recurrence_repair/core.py",
                    REPO_ROOT / "scripts/e01/run_s19_l06r.py",
                )
            ],
            "passed": True,
        },
    )
    (LOOP_ROOT / "repair_decision.md").write_text(
        "# S19-L06R Additive Repair Decision\n\n"
        "## Concise top summary\n\n"
        "- **Research step ID:** `S19-L06R`\n"
        "- **Completion status:** pre-outcome lock complete; execution pending\n"
        "- **Artifacts written:** repair preregistration, method/numerical contracts, immutable baseline, replay and benchmark evidence\n"
        "- **Validation result:** pre-outcome repository, immutable-input, frozen-clock/label, synthetic-fixture, and compute gates passed\n"
        "- **Outcome classification:** not yet accessed\n"
        "- **Caveats or blockers:** post-failure repair is adaptive; failed L06 remains immutable\n"
        "- **Recommended next action:** execute this one locked rerun, then mandatory human review\n\n"
        "Only the independent score equality policy changes: exact bit identity is replaced by the frozen S06 all-three numerical-equivalence rule. Every L06 scientific value, seed, label rule, control, statistic, and promotion gate remains unchanged.\n",
        encoding="utf-8",
    )
    append_root_ledgers(started.isoformat())
    write_json(
        LOOP_ROOT / "preparation_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l06r_preparation_runtime.v1",
            "startedUtc": started.isoformat(),
            "completedUtc": datetime.now(timezone.utc).isoformat(),
            "outcomeAccessed": False,
            "repositoryCommit": repository["head"],
            "passed": True,
        },
    )
    print(
        canonical_json(
            {
                "loopId": LOOP_ID,
                "status": "READY_FOR_LOCKED_EXECUTION",
                "repositoryCommit": repository["head"],
                "immutableFileCount": baseline["fileCount"],
                "replayedTrajectoryCount": replay["trajectoryCount"],
                "projectedTotalCpuHours": benchmark["projectedTotalCpuHours"],
            }
        )
    )


if __name__ == "__main__":
    main()
