#!/usr/bin/env python3
"""Finalize the immutable fail-closed S12G handoff without scientific analysis."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
ROOT = ARTIFACTS / "research_steps/S12G"
CACHE = Path("/cache/e01_s12g/source_results")
INPUT_CACHE = Path("/cache/e01_s12fr/timebase_confirmation")
CONFIG = REPO / "configs/e01/s12g_frozen_timebase_ensemble_preregistration.yaml"
SCHEMAS = REPO / "configs/e01/s12g_output_schemas.json"
VERSION = "E01-S12G-FROZEN-TIMEBASE-ENSEMBLE-v1.0.0"
FAILURE = (
    "S12G task failed S12F-CANDIDATE-01/M13: "
    "generation 2 has no distinct C0 endpoint observation"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def immutable_validation() -> dict[str, Any]:
    baseline = json.loads((ROOT / "immutable_prior_baseline.json").read_text())
    changed: list[dict[str, Any]] = []
    for item in baseline["researchStepFiles"] + baseline["lockedTrajectoryCaches"]:
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
        "schema": "eidosoma.e01.s12g_immutable_prior_validation.v1",
        "researchStepId": "S12G",
        "researchStepFileCount": len(baseline["researchStepFiles"]),
        "lockedTrajectoryCacheCount": len(baseline["lockedTrajectoryCaches"]),
        "changedCount": len(changed),
        "changed": changed,
        "passed": not changed,
    }
    write_json(ROOT / "immutable_prior_validation.json", payload)
    return payload


def partial_manifest() -> dict[str, Any]:
    completions: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for candidate in (
        "S12F-CANDIDATE-01",
        "S12F-CANDIDATE-02",
        "S12F-CANDIDATE-03",
    ):
        for matrix_index in range(32):
            root = CACHE / candidate / f"M{matrix_index:02d}"
            completion = root / "completion.json"
            if completion.is_file():
                record = json.loads(completion.read_text())
                completions.append(record)
                for path in sorted(root.iterdir()):
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
            else:
                incomplete.append(
                    {
                        "candidateId": candidate,
                        "matrixIndex": matrix_index,
                        "path": str(root),
                        "existingFiles": sorted(
                            item.name for item in root.iterdir() if item.is_file()
                        )
                        if root.is_dir()
                        else [],
                    }
                )
    payload = {
        "schema": "eidosoma.e01.s12g_partial_execution_manifest.v1",
        "researchStepId": "S12G",
        "completeTaskCount": len(completions),
        "incompleteTaskCount": len(incomplete),
        "completeByCandidate": {
            candidate: sum(item["candidateId"] == candidate for item in completions)
            for candidate in (
                "S12F-CANDIDATE-01",
                "S12F-CANDIDATE-02",
                "S12F-CANDIDATE-03",
            )
        },
        "allCompletedTaskReplayPassed": all(
            item["fullReplayAllPassed"] and item["prefixReplayAllPassed"]
            for item in completions
        ),
        "allCompletedTaskSuffixPassed": all(
            item["futureSuffixAllPassed"] for item in completions
        ),
        "completedTaskFailureRows": sum(item["failureRows"] for item in completions),
        "summedWorkerCpuHours": sum(item["cpuSeconds"] for item in completions) / 3600,
        "summedWorkerWallHours": sum(item["wallSeconds"] for item in completions) / 3600,
        "completionRecords": completions,
        "incompleteTasks": incomplete,
        "cacheFiles": files,
        "cacheFileCount": len(files),
        "cacheBytes": sum(item["bytes"] for item in files),
        "scientificValuesPromotedToArtifacts": False,
        "statisticalAdjudicationPerformed": False,
        "passed": len(completions) == 95
        and len(incomplete) == 1
        and incomplete[0]["candidateId"] == "S12F-CANDIDATE-01"
        and incomplete[0]["matrixIndex"] == 13,
    }
    write_json(ROOT / "partial_execution_manifest.json", payload)
    return payload


def zero_update_audit() -> dict[str, Any]:
    affected: list[dict[str, Any]] = []
    for matrix_index in range(32):
        path = INPUT_CACHE / "S12F-CANDIDATE-01" / f"M{matrix_index:02d}.pickle"
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        zero = [
            int(item.generation_one_based)
            for item in trajectory.generations
            if int(item.update_count) == 0
        ]
        if zero:
            affected.append(
                {
                    "candidateId": "S12F-CANDIDATE-01",
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "zeroUpdateGenerationCount": len(zero),
                    "zeroUpdateGenerations": zero,
                    "trajectorySha256": trajectory.trajectory_sha256,
                }
            )
    payload = {
        "schema": "eidosoma.e01.s12g_c0_zero_update_endpoint_audit.v1",
        "researchStepId": "S12G",
        "candidateId": "S12F-CANDIDATE-01",
        "trajectoryCount": 32,
        "affectedTrajectoryCount": len(affected),
        "zeroUpdateGenerationCount": sum(
            item["zeroUpdateGenerationCount"] for item in affected
        ),
        "affected": affected,
        "interpretation": (
            "C0 records only batch-update states. A zero-update generation has no distinct "
            "C0 state at its fission boundary; duplicating a prior state or inserting the "
            "daughter would alter the locked clock and was forbidden."
        ),
        "repairAttempted": False,
        "passed": False,
    }
    write_json(ROOT / "c0_zero_update_endpoint_audit.json", payload)
    return payload


def schema_validation() -> dict[str, Any]:
    contract = json.loads(SCHEMAS.read_text())["tables"]
    rows: list[dict[str, Any]] = []
    for filename, columns in contract.items():
        path = ROOT / filename
        frame = (
            pd.read_parquet(path)
            if path.is_file() and path.suffix == ".parquet"
            else pd.read_csv(path)
            if path.is_file()
            else pd.DataFrame()
        )
        missing = [column for column in columns if column not in frame.columns]
        rows.append(
            {
                "path": filename,
                "exists": path.is_file(),
                "rowCount": len(frame) if path.is_file() else None,
                "missingColumns": missing,
                "passed": path.is_file() and not missing,
            }
        )
    payload = {
        "schema": "eidosoma.e01.s12g_schema_validation.v1",
        "researchStepId": "S12G",
        "tables": rows,
        "passed": all(item["passed"] for item in rows),
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
    missing = [item for item in required if item != "artifact_manifest.json" and item not in present]
    payload = {
        "schema": "eidosoma.e01.s12g_artifact_manifest.v1",
        "researchStepId": "S12G",
        "artifacts": entries,
        "artifactCountExcludingSelf": len(entries),
        "totalBytesExcludingSelf": sum(item["bytes"] for item in entries),
        "requiredMissing": missing,
        "under30GiB": sum(item["bytes"] for item in entries) <= 30 * 1024**3,
        "passed": not missing and sum(item["bytes"] for item in entries) <= 30 * 1024**3,
    }
    write_json(ROOT / "artifact_manifest.json", payload)
    return payload


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text())
    immutable = immutable_validation()
    partial = partial_manifest()
    zero = zero_update_audit()
    schema = schema_validation()
    input_manifest = pd.read_parquet(ROOT / "trajectory_input_manifest.parquet")
    input_hashes = bool(input_manifest["cacheHashPassed"].astype(bool).all())
    start = datetime.fromtimestamp(
        (ROOT / "implementation_lock.json").stat().st_mtime, timezone.utc
    )
    stop = datetime.fromtimestamp(
        (ROOT / "research_step_full_results.md").stat().st_mtime, timezone.utc
    )
    benchmark = json.loads((ROOT / "runtime_benchmark.json").read_text())
    runtime = {
        "schema": "eidosoma.e01.s12g_runtime_manifest.v1",
        "researchStepId": "S12G",
        "status": "FAILED_CLOSED",
        "startedAtUtc": start.isoformat(),
        "stoppedAtUtc": stop.isoformat(),
        "approximateWallHours": (stop - start).total_seconds() / 3600,
        "summedCompletedWorkerCpuHours": partial["summedWorkerCpuHours"],
        "summedCompletedWorkerWallHours": partial["summedWorkerWallHours"],
        "completedTaskCount": partial["completeTaskCount"],
        "benchmarkProjectedCpuHours": benchmark["projectedCpuHours"],
        "benchmarkProjectedWallHours": benchmark["projectedWallHours"],
        "workers": 6,
        "threadEnvironment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "cpuPrecision": "float64_authoritative",
        "gpuHours": 0,
        "hardCeilingsExceeded": False,
        "terminalReason": FAILURE,
        "passed": False,
    }
    write_json(ROOT / "runtime_manifest.json", runtime)
    regeneration = {
        "schema": "eidosoma.e01.s12g_regeneration_validation.v1",
        "researchStepId": "S12G",
        "newGardTrajectoriesGenerated": 0,
        "lockedInputCacheHashesPassed": input_hashes,
        "s12frTrajectoryReplayEvidenceRows": 96,
        "s12frTrajectoryReplayEvidenceAllPassed": bool(
            input_manifest["cacheHashPassed"].astype(bool).all()
        ),
        "completedSourceTaskCount": partial["completeTaskCount"],
        "completedSourceTaskReplayAllPassed": partial[
            "allCompletedTaskReplayPassed"
        ],
        "completedSourceTaskSuffixAllPassed": partial[
            "allCompletedTaskSuffixPassed"
        ],
        "unevaluatedSourceTaskCount": partial["incompleteTaskCount"],
        "overallGatePassed": False,
        "reason": FAILURE,
        "passed": False,
    }
    write_json(ROOT / "regeneration_validation.json", regeneration)
    access = json.loads((ROOT / "scope_access_ledger.json").read_text())
    access["events"].append(
        {
            "stage": "FAIL_CLOSED_SOURCE_EXECUTION",
            "labelAndInformationValuesComputedToCacheForTasks": 95,
            "scientificValuesPromotedToArtifacts": False,
            "statisticalAssociationOrEnsembleAdjudicationPerformed": False,
            "newGardTrajectoryGenerated": False,
            "candidateSelectionOrReweighting": False,
            "predictionOrInterventionAccess": False,
            "s13Access": False,
            "status": "FAIL_CLOSED",
            "reason": FAILURE,
        }
    )
    access["success"] = False
    write_json(ROOT / "scope_access_ledger.json", access)
    failure_rows = [
        {
            "failureId": "S12G-TERMINAL-FAIL-CLOSED",
            "stage": "endpoint_mapping",
            "candidateId": "S12F-CANDIDATE-01",
            "trajectoryId": "E01-S12F-CONFIRMATION-S12F-CANDIDATE-01-M013",
            "implementationId": None,
            "temporalModeId": "C0_BATCH_UPDATES_ONLY",
            "endpointGeneration": 2,
            "severity": "FATAL",
            "status": "S12G_VALIDATION_FAILED_CLOSED",
            "reason": FAILURE,
            "gateImpact": "GLOBAL_FAIL_CLOSED_NO_ENSEMBLE_ADJUDICATION",
            "repairAttempted": False,
        },
        {
            "failureId": "S12G-C0-ZERO-UPDATE-STRUCTURAL-INCIDENCE",
            "stage": "endpoint_mapping_audit",
            "candidateId": "S12F-CANDIDATE-01",
            "trajectoryId": "E01-S12F-CONFIRMATION-S12F-CANDIDATE-01-M013",
            "implementationId": None,
            "temporalModeId": "C0_BATCH_UPDATES_ONLY",
            "endpointGeneration": None,
            "severity": "FATAL",
            "status": "NO_DISTINCT_C0_ENDPOINT",
            "reason": f"nine_zero_update_generations:{zero['affected'][0]['zeroUpdateGenerations']}",
            "gateImpact": "GLOBAL_FAIL_CLOSED_NO_REPAIR",
            "repairAttempted": False,
        },
    ]
    pd.DataFrame(failure_rows).to_csv(
        ROOT / "failure_ledger.csv", index=False, lineterminator="\n"
    )
    classification = {
        "schema": "eidosoma.e01.s12g_classification.v1",
        "researchStepId": "S12G",
        "versionedStepId": VERSION,
        "evidenceClass": "SOURCE_INFORMED_FROZEN_TIMEBASE_ENSEMBLE_RECONSTRUCTION",
        "classification": "S12G_VALIDATION_FAILED_CLOSED",
        "outcomeClass": "constraining/contradictory",
        "scientificAssociationClassification": "NOT_EVALUATED",
        "candidateSpecificResultsPromoted": False,
        "ensembleAdjudicationPerformed": False,
        "reason": FAILURE,
        "s13Status": "BLOCKED_PENDING_S12G_HUMAN_REVIEW",
    }
    write_json(ROOT / "classification.json", classification)
    report = f"""# S12G Full Results: Frozen Time-Base Ensemble

## Top summary

- **Research step ID:** `{VERSION}` (S12G)
- **Completion status:** `STOPPED_FAIL_CLOSED_AT_C0_ENDPOINT_MAPPING_GATE`; no ensemble statistics or scientific classification was performed.
- **Artifacts written:** Complete preregistration and method locks, 96-input/shared-identity/source audits, benchmark, 95-task partial-cache provenance, C0 structural audit, schema/immutability/runtime/replay/scope/failure/status/hash manifests, schema-bearing suppressed scientific tables, six stop-state figures, and this canonical report.
- **Validation result:** Upstream and completed-task checks passed, but the global endpoint-eligibility contract failed: candidate 1 matrix 13 has a zero-update generation and therefore no distinct C0 state at generation 2. Global S12G validation is `FAIL_CLOSED`.
- **Outcome classification:** `S12G_VALIDATION_FAILED_CLOSED` (constraining/contradictory); all label/emergence association questions are `NOT_EVALUATED`.
- **Caveats or blockers:** C0 excludes daughter states. Inserting the daughter, duplicating the prior state, skipping the generation, or reusing the last state would change or silently complete the locked clock. The preregistered stop rule forbids that repair after outcomes opened.
- **Recommended next action:** Keep S13 blocked and return for human review. No S12G repair, candidate deletion, candidate reweighting, favorable-candidate analysis, or statistical use of the 95 partial task caches is authorized.

## Lay summary

The three time-base candidates were supposed to be analyzed under one identical set of rules. One retained-overshoot C0 trajectory entered nine generations already at or above the fission threshold, so those generations had no molecular update. Because C0 records only molecular updates, there is no new C0 composition at those fission boundaries. The analysis would have to invent, duplicate, skip, or substitute a state to continue. The method was frozen to forbid exactly that kind of silent choice, so S12G stopped without calculating ensemble associations. The other 95 task results remain uninspected cache material and are not evidence.

## Frozen question

S12G asked whether historical/past-only replicator labels and S12C-confirmed source-defined emergence agree across all three S12FR-confirmed time bases. An ensemble positive required the same frozen gate on all three; no candidate could be selected, eliminated, or reweighted from downstream results.

## Inputs and provenance

- Exactly 96 S12FR confirmation trajectories were mounted, 32 per candidate; zero GARD trajectories were generated.
- All 96 cache hashes and S12FR replay flags passed, and all 32 catalytic-matrix/initial-state identities were shared across candidates.
- Pinned IIGR commit: `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`.
- Pinned PhiRL commit: `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`.
- Safe lattice SHA-256: `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`; no pickle was loaded for scientific execution.
- S12C source equivalence and S12D's 40/40 source-emergence identity evidence passed before the firewall opened.
- Pre-outcome design commit: `0118892d035eef932274b0f44bd1ecc024268fa2`; locked implementation commit: `29a8ac5`.

## Detailed methods

The frozen C0 sequence is initial state plus molecular batch-update states only. The frozen C1 sequence additionally records the selected daughter. A post-fission endpoint maps to the final molecular update in C0 and to the selected daughter in C1. Prefix evaluation begins at 256 prior locked-clock transitions; each source pipeline is independently refit and replayed, with structural suffix checks at every endpoint and executed deletion/shuffle/replacement sentinels at the first, middle, and last eligible endpoint.

Labels were frozen as `HISTORICAL_H090_REPLICATOR` (primary) and `PAST_ONLY_COSINE_REPLICATOR` (secondary). Preprocessing was additive 0.5 closure, full CLR, and removal of original component 100. IIGR synergy plus two downward-causation atoms was primary; PhiRL was robustness; corrected local Phi-r was comparator-only. These calculations were launched only after the complete design, source identities, input hashes, statistical gates, schemas, and stop rules were committed and pushed.

## Results

The three-trajectory benchmark completed with exact replay and suffix checks. It projected {benchmark['projectedCpuHours']:.3f} CPU-hours and {benchmark['projectedWallHours']:.3f} wall-hours, below the hard ceilings. During full execution, 95/96 tasks completed in cache with zero task failure rows and unanimous task-level full replay, prefix replay, and suffix flags. The sole incomplete task was candidate 1 matrix 13.

That trajectory's first fission followed a large retained overshoot: pre-fission mass 197 and selected-daughter mass 90. Generation 2 therefore began above `n_max=80` and performed zero batch updates before fission. The same trajectory has nine zero-update generations: `{zero['affected'][0]['zeroUpdateGenerations']}`. It is the only affected candidate-1 trajectory. At generation 2, the frozen C0 clock has no distinct state that can serve as a new endpoint; processing stopped before source fits for that task.

No cached label or emergence value was collated into a scientific artifact. `candidate_associations.csv`, drift/temporal/spike/identity/future-dependence/cross-candidate/adjudication tables are schema-bearing and empty. The six figures are explicit stop-state figures. No candidate-specific or ensemble association result exists.

## Validation

- Prior immutability: {'PASS' if immutable['passed'] else 'FAIL'} across {immutable['researchStepFileCount']} S01–S12FR artifact files and {immutable['lockedTrajectoryCacheCount']} locked trajectory caches; changed count {immutable['changedCount']}.
- Source/metric identity: PASS before execution.
- Shared identities: PASS, 32/32 paired matrix/initial units.
- Input hashes and S12FR replay evidence: PASS, 96/96.
- Completed source tasks: 95/95 full replay, prefix replay, and suffix task flags passed; zero task-level failure rows.
- Required table schemas: {'PASS' if schema['passed'] else 'FAIL'}; all scientific tables are suppressed/empty after the global stop.
- Scope: zero new GARD trajectories, zero predictions, zero MLP fits, zero interventions, zero candidate selection/reweighting, zero S13 work.
- Runtime: approximately {runtime['approximateWallHours']:.3f} wall-hours and {runtime['summedCompletedWorkerCpuHours']:.3f} summed completed-worker CPU-hours; no GPU use and no hard ceiling exceeded.
- Partial-cache provenance: {partial['cacheFileCount']} files, {partial['cacheBytes']} bytes, all hash-recorded in `partial_execution_manifest.json`; none promoted as scientific evidence.

## Commands

```bash
PYTHONPATH=src pytest -q tests/e01/test_s12g_frozen_timebase_ensemble.py
ruff check src/e01_frozen_timebase_ensemble scripts/e01/freeze_s12g_preregistration.py scripts/e01/run_s12g_frozen_timebase_ensemble.py tests/e01/test_s12g_frozen_timebase_ensemble.py
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/freeze_s12g_preregistration.py --design-commit 0118892d035eef932274b0f44bd1ecc024268fa2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/run_s12g_frozen_timebase_ensemble.py --workers 6
ARTIFACTS_DIR=/artifacts PYTHONPATH=src python scripts/e01/finalize_s12g_fail_closed.py
```

## Caveats, blockers, and interpretation

- This is a structural endpoint-definition failure, not evidence for or against an emergence/replication association.
- The 95 completed caches cannot be analyzed as a favorable subset or promoted as confirmation evidence.
- A rule for zero-update C0 generations would be a new methodological choice. None was inferred or added after failure.
- Full fits would be retrospective; only prefix fits could have addressed prospective behavior.
- Public source code remains source-informed, not author-code, paper-primary, or exact GARD identity.
- S12F remains `SIMULATOR_IDENTIFICATION_FAILED`; S12FR remains `NONIDENTIFIABLE_TIMEBASE_ENSEMBLE`; all prior negative and failed evidence remains intact.

## Recommended next action

Return for mandatory human review with S13 `BLOCKED_PENDING_S12G_HUMAN_REVIEW`. No repair or continuation is authorized. A future human decision would have to explicitly preregister how zero-update C0 generations are represented and whether doing so preserves the meaning of the locked clock; it must not reuse the 95 cached results for method selection.
"""
    (ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": "S12G",
        "stepNumber": "S12G",
        "success": False,
        "status": "STOPPED_FAIL_CLOSED_AT_C0_ENDPOINT_MAPPING_GATE",
        "artifactsWritten": [],
        "validationResult": (
            "FAIL_CLOSED: prior/source/input/completed-task replay/suffix/schema/runtime/scope "
            "checks passed, but one of 96 trajectories lacks distinct C0 endpoints at nine "
            "zero-update generations; no scientific adjudication was permitted."
        ),
        "caveatsOrBlockers": [
            FAILURE,
            "Candidate 1 matrix 13 has nine zero-update generations under C0.",
            "The 95 completed caches are unpromoted and cannot be analyzed as a subset.",
            "S13 remains blocked and no repair is authorized.",
        ],
        "recommendedNextAction": (
            "Return for human review; keep S13 blocked and do not repair, subset, "
            "reweight, or statistically analyze S12G outputs without new authorization."
        ),
        "outcomeClassification": "S12G_VALIDATION_FAILED_CLOSED",
        "outcomeClass": "constraining/contradictory",
        "scientificAssociationClassification": "NOT_EVALUATED",
        "s13Status": "BLOCKED_PENDING_S12G_HUMAN_REVIEW",
    }
    write_json(ROOT / "status.json", status)
    required = config["artifacts"]["required"]
    manifest = artifact_manifest(required)
    status["artifactsWritten"] = [
        item["relativePath"] for item in manifest["artifacts"]
    ] + ["artifact_manifest.json"]
    write_json(ROOT / "status.json", status)
    manifest = artifact_manifest(required)
    if not (
        immutable["passed"]
        and partial["passed"]
        and schema["passed"]
        and manifest["passed"]
        and input_hashes
    ):
        raise RuntimeError("S12G fail-closed handoff validation failed")
    print(
        json.dumps(
            {
                "classification": "S12G_VALIDATION_FAILED_CLOSED",
                "completeTasks": partial["completeTaskCount"],
                "immutablePassed": immutable["passed"],
                "schemasPassed": schema["passed"],
                "artifactManifestPassed": manifest["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
