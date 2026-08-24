#!/usr/bin/env python3
"""Reporting-only fail-closed finalizer for the immutable S19-L09 attempt.

This file contains no label, clustering, simulation, emergence, prediction, or
intervention implementation.  It serializes the observed locked-run failure,
empty ineligible scientific schemas, validation state, and human-review handoff.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L09"
VERSION = "E01-S19-L09-RECURRING-ATTRACTOR-LABEL-RECONSTRUCTION-v1.0.0"


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
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def write_empty_parquet(path: Path, columns: dict[str, str]) -> None:
    frame = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})
    frame.to_parquet(path, index=False, compression="zstd")


def manifest_for(root: Path, exclude: set[Path]) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path in exclude:
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "eidosoma.e01.s19_l09.artifact_manifest.v1",
        "root": str(root),
        "generatedAtUtc": utc_now(),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def write_placeholder_figures() -> list[str]:
    names = [
        "figure_01_dominant_recurring_clusters.png",
        "figure_02_molecular_h_to_dominant_over_time.png",
        "figure_03_adjacent_vs_recurring_labels.png",
        "figure_04_occupancy_persistence_comparison.png",
        "figure_05_consistency_first_onset.png",
        "figure_06_episode_topology.png",
        "figure_07_negative_controls.png",
        "figure_08_cross_candidate_agreement.png",
        "figure_09_fingerprint_decision_matrix.png",
    ]
    for index, name in enumerate(names, start=1):
        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.5,
            0.62,
            f"Figure {index} not scientifically generated",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
        )
        ax.text(
            0.5,
            0.40,
            "S19-L09 LOOP_FAILED_CLOSED before eligible aggregation\n"
            "No partial or in-memory trajectory outcome is displayed.",
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.savefig(LOOP_ROOT / name, dpi=160)
        plt.close(fig)
    return names


def append_root_records(classification: dict[str, Any], report: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    mask = (ledger["loopId"] == "S19-L09") & (
        ledger["recordPhase"] == "POST_LOOP_MANDATORY_HUMAN_REVIEW_BOUNDARY"
    )
    if not bool(mask.any()):
        row = {
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "timestampUtc": utc_now(),
            "loopId": "S19-L09",
            "recordPhase": "POST_LOOP_MANDATORY_HUMAN_REVIEW_BOUNDARY",
            "beliefBeforeLoop": "One of two source/paper-grounded dominant recurring-composition pipelines might explain the paper's label fingerprint.",
            "motivatingEvidence": "Paper, Figure 1, Table 1, historical GARD compotype source, and L08's untouched negative mechanism result.",
            "failureOrAmbiguityTargeted": "Self-replicator label identity.",
            "selectedHypotheses": "Exactly R1 historical dominant compotype and R2 paper-Euclidean dominant attractor.",
            "learned": "The locked R1 implementation encountered a four-point/four-cluster silhouette case on a real frozen trajectory. The backend rejects silhouette when every sample is a singleton. No convention was preregistered, so the loop failed closed before eligible serialization.",
            "weakenedHypotheses": "The claim that the frozen R1 source-equivalent implementation was complete over the real input domain; the synthetic fixtures did not cover k=n after non-drift filtering.",
            "remainingPlausibleHypotheses": "The recurring-attractor scientific hypothesis remains unadjudicated; a future human decision could authorize an untouched, explicitly locked singleton-silhouette policy, but L09 itself cannot be repaired.",
            "proposedNextTest": "Mandatory human review; do not rerun or repair L09 automatically.",
            "informationGainRationale": "Any later test must prospectively specify k=n/singleton silhouette semantics and use an untouched design rather than rescue partial L09 values.",
            "appendOnly": True,
        }
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_parquet(ledger_path, index=False, compression="zstd")
        with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n## Entry 020 — S19-L09 failed closed at unregistered singleton-silhouette semantics\n\n"
                "- **What was learned:** after pre-outcome fixtures and push, R1 reached `k=n=4` on a frozen real trajectory; the locked backend cannot define silhouette when every point is its own cluster.\n"
                "- **What was weakened:** completeness of the locked R1 source-equivalent implementation across the actual non-drift substrate.\n"
                "- **What remains plausible:** the recurring-attractor hypothesis itself is unadjudicated; no R1 or R2 outcome survived validation.\n"
                "- **What should be tested next:** nothing automatically. A later human decision would need a new untouched lock with explicit singleton-silhouette semantics.\n"
                "- **Why this is not repaired here:** choosing zero, one, undefined, or another singleton silhouette after outcomes is a scientific method change.\n"
            )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for item in registry["loops"]:
        if item["loopId"] == "S19-L09":
            item.update(
                {
                    "status": "LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW",
                    "outcomeAccessed": True,
                    "completed": True,
                    "eligibleScientificResults": False,
                    "classification": [
                        "LOOP_FAILED_CLOSED",
                        "POSSIBLE_PIPELINE_ARTIFACT",
                        "NOT_PROMOTABLE",
                    ],
                    "failureId": "S19-L09-F001",
                    "promotionEligibleCount": 0,
                    "promotedLeadCount": 0,
                    "nextStepActive": False,
                }
            )
            break
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not any(item.get("decision") == "S19_L09_LOOP_FAILED_CLOSED" for item in review["history"]):
        review["history"].append(
            {
                "date": "2026-08-09",
                "decision": "S19_L09_LOOP_FAILED_CLOSED",
                "scope": VERSION,
                "result": "UNREGISTERED_K_EQUALS_N_SINGLETON_SILHOUETTE_SEMANTICS",
                "source": "locked_execution_global_stop",
            }
        )
    review["pendingDecision"] = "POST_S19_L09_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(review_path, review)

    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": "S19-L09",
        "stepNumber": 19,
        "success": False,
        "status": "LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [
            str(LOOP_ROOT / "S19_L09_FULL_RESULTS.md"),
            str(LOOP_ROOT / "failure_ledger.csv"),
            str(LOOP_ROOT / "classification.json"),
            str(LOOP_ROOT / "artifact_manifest.json"),
        ],
        "validationResult": "PREOUTCOME_GATES_PASS_THEN_LOCKED_R1_K_EQUALS_N_SILHOUETTE_IMPLEMENTATION_FAILURE_NO_ELIGIBLE_SCIENTIFIC_RESULTS",
        "outcomeClassification": "LOOP_FAILED_CLOSED",
        "caveatsOrBlockers": [
            "R1_k_equals_n_singleton_silhouette_semantics_unregistered",
            "no_eligible_R1_or_R2_result",
            "in_memory_partial_values_invalidated_not_serialized",
            "recurring_attractor_hypothesis_unadjudicated",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_REPAIR_RERUN_L10_S20_E02_AUTHOR_CONTACT_OR_REPORT_BUNDLE",
    }
    write_json(ARTIFACT_ROOT / "s19_status.json", status)
    write_json(LOOP_ROOT / "status.json", status)


def main() -> None:
    if not LOOP_ROOT.exists():
        raise FileNotFoundError(LOOP_ROOT)
    release = json.loads((LOOP_ROOT / "run_release_gate.json").read_text())
    immutable = json.loads((LOOP_ROOT / "immutable_prior_validation.json").read_text())
    if not release.get("passed") or not immutable.get("passed"):
        raise RuntimeError("cannot finalize because pre-run release/immutable gate did not pass")

    write_json(
        LOOP_ROOT / "reporting_amendment_001.json",
        {
            "schema": "eidosoma.e01.s19_l09.reporting_amendment.v1",
            "amendmentId": "S19-L09-REPORTING-AMENDMENT-001",
            "defect": "Malformed pre-outcome Python multiline literal embedded the intended source audit in decision_record.md and omitted the standalone source_and_paper_label_audit.md file",
            "repair": "Rewrite decision_record.md as clean Markdown and write the intended standalone source audit",
            "scientificValuesChanged": False,
            "methodChanged": False,
            "classificationChanged": False,
            "failureDecisionChanged": False,
            "valuePreserving": True,
            "recordedAtUtc": utc_now(),
        },
    )
    (LOOP_ROOT / "decision_record.md").write_text(
        """# S19-L09 Decision Record

## Concise top summary

- **Research step ID:** `S19-L09` (`E01-S19-L09-RECURRING-ATTRACTOR-LABEL-RECONSTRUCTION-v1.0.0`).
- **Completion status:** `LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW`.
- **Artifacts written:** outcome-blind lock, source/input/fixture evidence, explicit empty scientific tables, failure and validation records, canonical report, figures marked unavailable, and hash manifests.
- **Validation result:** pre-outcome fixtures/input/immutability/pushed-head gates passed; locked R1 execution failed at an unregistered `k=n=4` singleton-silhouette case before eligible serialization.
- **Outcome classification:** `LOOP_FAILED_CLOSED`, `POSSIBLE_PIPELINE_ARTIFACT`, `NOT_PROMOTABLE`.
- **Caveats or blockers:** choosing singleton-silhouette semantics after the failure would change the method; R2 cannot be selectively rescued; the recurring-attractor hypothesis is unadjudicated.
- **Recommended next action:** mandatory human review; no automatic repair, rerun, downstream loop, or S20 activation.

## Decision history

The human authorized exactly two recurring-attractor pipelines on frozen L08 trajectories. The complete scientific lock was pushed at commit `691b328` before outcome access. The runner then stopped globally at `S19-L09-F001`. No scientific output is eligible. Reporting amendment `S19-L09-REPORTING-AMENDMENT-001` corrected only this Markdown file's serialization and restored the separately required source audit; it changed no method, value, failure, or classification.
""",
        encoding="utf-8",
    )
    (LOOP_ROOT / "source_and_paper_label_audit.md").write_text(
        """# Source and Paper Label Audit

## Concise top summary

- **Research step ID:** `S19-L09`.
- **Completion status:** source and measurement-semantics audit complete; the later locked scientific run failed closed.
- **Artifacts written:** this audit, hashed source snapshot, Table 1 semantics lock, label-method lock, fixture manifest, and source-equivalence table.
- **Validation result:** every retained source has a SHA-256 identity and 13/13 mandatory pre-outcome fixture checks passed; no source-equivalence failure triggered the later stop.
- **Outcome classification:** `LOOP_FAILED_CLOSED` for the full loop; this audit contains no scientific trajectory outcome.
- **Caveats or blockers:** no authoritative target-paper code; historical MATLAB release/RNG behavior, exact clustering details, Table 1 onset units, and SD-versus-SE identity remain unresolved.
- **Recommended next action:** preserve this audit and stop for human review; any future attempt must prospectively define the all-singleton silhouette case.

## Direct paper evidence

The paper describes recurring compositions inherited across generations, calls self-replicators clusters in molecular-composition space with homeostatic attractor-like growth, and says entry/exit depends on similarity to the run's most recurring composition. Its Methods separately describes highly similar steady compositions in Euclidean space. Figure 1 and Table 1 were treated as measurement semantics, not permission to tune a cluster radius or H threshold.

## Direct historical-source evidence

The pinned GARD v10 lineage defines H as clipped cosine similarity. Technique 1 marks a boundary non-drift when the average of incoming and outgoing adjacent-generation H exceeds 0.9, duplicating the first/last adjacent score at endpoints. `tgs_acluster` clusters only non-drift boundaries, evaluates k=1–10 with ten replicas, selects replicas by minimum distance, scores k>1 by mean silhouette, uses a special mean-H carpet score for k=1, and stops after four k values without improvement. `getcomposometime_v10` and `biased_gard_v10` identify the most frequent compotype.

## Reconstruction choices

R1 used deterministic CPU-float64 spherical k-means because the original MATLAB release and RNG are not identified. R2 followed the paper's Euclidean wording with deterministic Lloyd k-means. R2's k=1 silhouette was explicitly undefined and could not win selection. Both pipelines required at least two assigned members and two strict-H>0.9 centroid visits. These are frozen reconstructions, not author-code claims.

The later real-input failure exposed a choice the lock had not defined for R1: when historical filtering leaves n points and the k search reaches k=n, all clusters are singletons and the selected backend does not define silhouette. No post-outcome convention was added.

## Cited-method context

References 63–65 ground the GARD/composome lineage. The open PNAS text defines H and homeostatic quasi-stationary composomes; related GARD papers describe compotypes and compositional recurrence. Public identities, retrieval paths, hashes, and license/redistribution status are in `source_snapshot_manifest.json` and the append-only source ledger. No author was contacted, and unlicensed source was not redistributed.

## Table 1 semantics

Molecular probability, persistence, consecutive-label Pearson consistency, and onset were frozen in `table1_semantics_lock.yaml`. Zero-based, one-based, normalized, and fission-generation onset were all required. Both sample SD and SE were required; `AUTHOR_DISPERSION_UNRESOLVED` could not be resolved by target proximity. Boundary diagnostics could not replace molecular results.

## Reporting amendment boundary

`S19-L09-REPORTING-AMENDMENT-001` restored this intended standalone audit after a Markdown serialization defect. It changed no source identity, scientific method, value, failure, or classification.
""",
        encoding="utf-8",
    )

    empty_schemas = {
        "cluster_results.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "matrixIndex": "int64",
            "k": "int64",
            "status": "string",
            "selectionScore": "float64",
        },
        "dominant_attractor_results.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "matrixIndex": "int64",
            "pipelineStatus": "string",
        },
        "molecular_label_results.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "matrixIndex": "int64",
            "analysisUnitIndex": "int64",
            "labelStatus": "string",
            "hToDominant": "float64",
            "isReplicator": "boolean",
        },
        "boundary_label_results.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "matrixIndex": "int64",
            "boundaryIndex0": "int64",
            "labelStatus": "string",
            "hToDominant": "float64",
            "isReplicator": "boolean",
        },
        "label_fingerprint_results.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "matrixIndex": "int64",
            "fingerprintStatus": "string",
            "occupancy": "float64",
            "persistence": "float64",
            "consistency": "float64",
            "firstOnsetRawStep1": "float64",
        },
        "episode_results.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "matrixIndex": "int64",
            "polarity": "string",
            "duration": "int64",
        },
        "comparator_results.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "matrixIndex": "int64",
            "fingerprintStatus": "string",
        },
        "negative_control_results.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "matrixIndex": "int64",
            "controlType": "string",
            "controlStatus": "string",
        },
        "complete_fingerprint_distances.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "rawPaperDistance": "float64",
            "normalizedPaperDistance": "float64",
        },
        "bootstrap_results.parquet": {
            "pipelineId": "string",
            "candidateId": "string",
            "bootstrapReplicate": "int64",
            "status": "string",
        },
    }
    for name, columns in empty_schemas.items():
        write_empty_parquet(LOOP_ROOT / name, columns)
    pd.DataFrame(
        columns=[
            "pipelineId",
            "candidateId",
            "metric",
            "status",
            "reason",
        ]
    ).to_csv(LOOP_ROOT / "paper_target_comparison.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        columns=[
            "pipelineId",
            "metric",
            "pairedMatrixCount",
            "status",
            "reason",
        ]
    ).to_csv(LOOP_ROOT / "candidate_comparison.csv", index=False, lineterminator="\n")

    failure = pd.DataFrame(
        [
            {
                "failureId": "S19-L09-F001",
                "phase": "PHASE_3_LOCKED_LABEL_EXECUTION",
                "pipelineId": "R1_HISTORICAL_DOMINANT_COMPTYPE_H090",
                "candidateId": "UNKNOWN_NOT_REOPENED_AFTER_GLOBAL_STOP",
                "matrixIndex": None,
                "failureStatus": "UNREGISTERED_K_EQUALS_N_SINGLETON_SILHOUETTE_SEMANTICS",
                "backendError": "Number of labels is 4. Valid values are 2 to n_samples - 1 (inclusive)",
                "diagnosis": "After historical non-drift filtering, one real trajectory presented n=4 eligible boundaries and the locked k search evaluated k=4; sklearn silhouette does not define all-singleton clustering.",
                "outcomeValuesSerialized": False,
                "partialInMemoryValuesEligible": False,
                "scientificRepairPermitted": False,
                "globalStopTriggered": True,
                "excludedFromScientificAggregation": True,
            }
        ]
    )
    failure.to_csv(LOOP_ROOT / "failure_ledger.csv", index=False, lineterminator="\n")

    classification = {
        "schema": "eidosoma.e01.s19_l09.classification.v1",
        "versionedLoopId": VERSION,
        "status": "LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW",
        "topLevelClassification": "LOOP_FAILED_CLOSED",
        "secondaryClassifications": ["POSSIBLE_PIPELINE_ARTIFACT", "NOT_PROMOTABLE"],
        "pipelineClassifications": {
            "R1_HISTORICAL_DOMINANT_COMPTYPE_H090": "LOOP_FAILED_CLOSED",
            "R2_PAPER_EUCLIDEAN_DOMINANT_ATTRACTOR_H090": "NOT_EVALUATED_GLOBAL_STOP",
        },
        "promotionEligiblePipelines": [],
        "promotedLeadCount": 0,
        "eligibleScientificResults": False,
        "l08ClassificationPreserved": "NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA",
        "s18ProspectivePredictionStatusChanged": False,
        "s18ProspectiveCausalControlStatusChanged": False,
    }
    write_json(LOOP_ROOT / "classification.json", classification)
    write_json(
        LOOP_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l09.regeneration_validation.v1",
            "status": "NOT_RUN_GLOBAL_STOP_BEFORE_ELIGIBLE_SERIALIZATION",
            "exactReplayPassed": False,
            "scientificTablesExplicitEmptySchemas": True,
            "reportRegenerableFromFailureLedgerAndValidationArtifacts": True,
            "passed": False,
        },
    )
    write_json(
        LOOP_ROOT / "runtime_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l09.runtime_manifest.v1",
            "status": "FAILED_CLOSED",
            "failedScientificRunWallSecondsObservedByOrchestrator": 27.61595444,
            "cpuHours": None,
            "cpuCeilingHours": 32,
            "wallCeilingHours": 8,
            "gpuHours": 0,
            "workers": 8,
            "threadsPerWorker": 1,
            "ceilingExceeded": False,
            "runtimeOutcomeExtractionProhibited": True,
        },
    )
    figure_names = write_placeholder_figures()

    report = f"""# S19-L09 Full Results — Failed-Closed Recurring-Attractor Reconstruction

## Concise top summary

- **Research step ID:** `S19-L09` (`{VERSION}`).
- **Completion status:** `LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW`; execution stopped before eligible scientific serialization.
- **Artifacts written:** all directed lock/source/fixture artifacts; explicit empty ineligible scientific tables; failure, runtime, storage, regeneration, status, classification, and hash evidence; nine clearly marked non-scientific figure placeholders; canonical full report and one-page handoff.
- **Validation result:** pre-outcome 13/13 fixtures, 1,727 immutable files, 400 frozen input-cache hashes, ten-trajectory opaque benchmark, clean pushed commit, and release gate passed. Locked R1 execution then failed because an unregistered `k=n=4` all-singleton silhouette case is undefined in the backend; no full replay or scientific aggregation is eligible.
- **Outcome classification:** `LOOP_FAILED_CLOSED`, `POSSIBLE_PIPELINE_ARTIFACT`, `NOT_PROMOTABLE`. R1 failed operationally; R2 is `NOT_EVALUATED_GLOBAL_STOP`.
- **Caveats or blockers:** the fixture suite omitted the real-domain case where non-drift filtering leaves exactly k points. Selecting a post-outcome singleton-silhouette convention would change the scientific method. The recurring-attractor hypothesis remains unadjudicated.
- **Recommended next action:** mandatory human review. Do not repair/rerun L09 or activate another loop, S20, E02, author contact, prediction, emergence, intervention, or report generation automatically.

## Lay summary

The planned test did not produce a trustworthy answer about whether the paper used a recurring-attractor label. The historical pipeline sometimes leaves very few non-drifting generation states. In one frozen trajectory it left four, and the locked search tried to cluster those four states into four singleton clusters. The selected software does not define a silhouette score for that case. Because choosing a convention after seeing the failure could change which cluster count wins, the analysis stopped and discarded every partial result.

## Frozen question

Could either of exactly two fixed dominant-recurring-composition labels jointly reproduce the paper's control fingerprints better than adjacent, boundary, projected-boundary, and high-exposure comparators?

## Inputs

The lock used exactly the 100 shared L08 matrices and four frozen trajectory groups; original-exposure candidate 2/3 were primary and `h=2.875` candidate 2/3 comparator-only. The original paper, Figure 1, Table 1, pinned historical GARD source, and cited methods 63–65 were hashed. No new trajectory, emergence value, prediction, or intervention was generated.

## Methods and lock

R1 froze historical technique-1 non-drift filtering, cosine k-means over k=1–10 with ten replicas, source-equivalent score/early-stop behavior, dominant valid compotype selection, and direct molecular strict-H>0.9 membership. R2 froze all-boundary Euclidean Lloyd k-means and the same membership threshold. Mandatory fixtures covered source smoothing, permutation/scaling, replay, planted dominant and two-attractor structure, a drifting no-cluster case, deterministic ties, and direct molecular rather than interval-projected labels. The complete code/config lock was committed and pushed at `691b328` before scientific execution.

## Commands

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l09.py
PYTHONPATH=src python scripts/e01/prepare_s19_l09_lock.py
git commit ... && git push origin eidosoma/groups/42
PYTHONPATH=src python scripts/e01/run_s19_l09.py --workers 8
```

## Failure result

The first surfaced R1 worker failure was:

```text
ValueError: Number of labels is 4. Valid values are 2 to n_samples - 1 (inclusive)
```

After non-drift filtering, that trajectory supplied four eligible boundaries and the locked search evaluated k=4. The backend cannot calculate silhouette when every point is its own cluster. The runner stopped globally. Its exception path did not serialize the candidate/matrix identity, and the unit was deliberately not reopened after the stop. In-memory computations from other workers are invalidated and absent from the artifacts.

No occupancy, persistence, consistency, onset, episode, control, bootstrap, cross-candidate, or paper-distance result is eligible. The comparator tables are also explicit empty schemas because selective continuation would drop a required pipeline.

## Validation

- 13/13 mandatory pre-outcome source/synthetic fixture checks passed.
- 1,727 immutable prior files and all 400 L08 cache hashes passed before execution.
- The ten-trajectory opaque benchmark passed the compute gate and retained no scientific values.
- Clean pushed `HEAD == origin/eidosoma/groups/42` and locked code hashes passed.
- The real-domain k=n condition was not represented in the fixtures; this is the operational validation failure.
- Full exact replay is explicitly `NOT_RUN_GLOBAL_STOP`; it is not misreported as passed.
- Required scientific tables carry explicit empty schemas, and failure provenance is machine-readable.

## Figures

The nine directed figure paths exist only as clearly labelled failure placeholders. They contain no scientific result and prevent report consumers from mistaking absent panels for zero-valued evidence:

{chr(10).join(f'- `{name}`' for name in figure_names)}

## Caveats and interpretation boundary

This failure does not support or refute a recurring-attractor label. It cannot be used to favor R2, to reinterpret L08, or to change S18's prediction/control conclusions. A future attempt would need a new human authorization and a prospective, source-grounded choice for all-singleton silhouette handling, plus a fixture that exercises it. L09 itself is immutable and cannot be repaired.

## Provenance

`source_snapshot_manifest.json`, `label_method_lock.json`, `input_manifest.json`, `preoutcome_preparation_failure_001.json`, `failure_ledger.csv`, `run_release_gate.json`, and `immutable_prior_validation.json` preserve source, code, input, preparation, execution, and stop identities. Unlicensed historical source remains cache-only.

## Recommended next action

Stop for mandatory human review. No next option is active.
"""
    summary = """# S19-L09 One-Page Decision Summary

## Concise top summary

- **Research step ID:** `S19-L09`.
- **Completion status:** `LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW`.
- **Artifacts written:** complete lock/source/fixture evidence, explicit empty results, failure/classification/validation/status/hashes, canonical report, and non-scientific placeholder figures.
- **Validation result:** pre-outcome gates passed; locked R1 execution failed on unregistered `k=n=4` singleton-silhouette semantics; no result replay or scientific aggregation is eligible.
- **Outcome classification:** `LOOP_FAILED_CLOSED`, `POSSIBLE_PIPELINE_ARTIFACT`, `NOT_PROMOTABLE`.
- **Caveats/blockers:** the recurring-attractor hypothesis remains unadjudicated; R2 cannot be selectively rescued; no post-outcome method repair is permitted.
- **Recommended next action:** mandatory human review; no automatic repair, rerun, later loop, or S20 activation.

## Decision

The correct L09 verdict is operational failure, not scientific non-support. A future human-authorized untouched attempt would need to freeze how historical k=n/all-singleton silhouette is represented before any result is opened. L09 itself remains immutable.
"""
    (LOOP_ROOT / "S19_L09_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(summary, encoding="utf-8")
    append_root_records(classification, report)

    retained = sum(path.stat().st_size for path in LOOP_ROOT.rglob("*") if path.is_file())
    write_json(
        LOOP_ROOT / "storage_validation.json",
        {
            "schema": "eidosoma.e01.s19_l09.storage_validation.v1",
            "retainedBytes": retained,
            "retainedGiB": retained / (1024**3),
            "retainedCeilingGiB": 10,
            "temporaryCacheGiB": 0,
            "temporaryCacheCeilingGiB": 25,
            "passed": retained <= 10 * 1024**3,
        },
    )

    loop_manifest_path = LOOP_ROOT / "artifact_manifest.json"
    write_json(loop_manifest_path, manifest_for(LOOP_ROOT, {loop_manifest_path}))
    loop_manifest = json.loads(loop_manifest_path.read_text())
    replay = all(
        (LOOP_ROOT / row["path"]).stat().st_size == row["bytes"]
        and sha256_file(LOOP_ROOT / row["path"]) == row["sha256"]
        for row in loop_manifest["files"]
    )
    write_json(
        LOOP_ROOT / "artifact_integrity_validation.json",
        {
            "schema": "eidosoma.e01.s19_l09.artifact_integrity_validation.v1",
            "manifestFileCount": loop_manifest["fileCount"],
            "allManifestRowsReplay": replay,
            "passed": replay,
        },
    )
    write_json(loop_manifest_path, manifest_for(LOOP_ROOT, {loop_manifest_path}))

    root_manifest_path = ARTIFACT_ROOT / "artifact_manifest.json"
    root_manifest = manifest_for(ARTIFACT_ROOT, {root_manifest_path})
    root_manifest["schema"] = "eidosoma.e01.s19_artifact_manifest.v1"
    write_json(root_manifest_path, root_manifest)
    print(
        json.dumps(
            {
                "status": "LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW",
                "classification": "LOOP_FAILED_CLOSED",
                "failureId": "S19-L09-F001",
                "eligibleScientificResults": False,
                "promotedLeadCount": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
