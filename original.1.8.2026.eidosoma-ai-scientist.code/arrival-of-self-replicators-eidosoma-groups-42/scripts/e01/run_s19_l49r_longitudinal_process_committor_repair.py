#!/usr/bin/env python3
"""Run the additive S19-L49R outcome-blind state-availability repair."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L49 = load_module(
    "e01_l49r_frozen_l49_runner",
    ROOT / "scripts/e01/run_s19_l49_longitudinal_process_committor.py",
)

FAILED_L49_ROOT = Path("/artifacts/research_steps/S19/loops/L49")
LOOP_ROOT = Path("/artifacts/research_steps/S19/loops/L49R")
BUILD_ROOT = Path("/cache/e01_s19_l49r/build")
CONFIG = ROOT / "configs/e01/s19_l49r_longitudinal_process_committor_repair.yaml"
RUNNER_PATH = Path(__file__).resolve()
LOOP_ID = "S19-L49R"
VERSION = "E01-S19-L49R-LANDMARK-AVAILABILITY-REPAIR-v1.0.0"
FAILED_L49_VERSION = "E01-S19-L49-LONGITUDINAL-PROCESS-COMMITTOR-RISK-TRAJECTORY-v1.0.0"
SEED_ROOT = bytes.fromhex(
    "661d7f3fe4a51523c427899e657d88455d6861dafd67fe3e5ee9af5026b51c14"
)

ORIGINAL_VALIDATE = L49.validate_immutable_prior
ORIGINAL_SOURCE_REGISTRY = L49.source_registry
ORIGINAL_REPORT = L49.report_text
ORIGINAL_MANIFEST = L49.manifest_for
ORIGINAL_PREPARE = L49.prepare_lock
ORIGINAL_EXECUTE = L49.execute


def _sha256_file(path: Path) -> str:
    return L49.sha256_file(path)


def validate_immutable_prior() -> dict[str, Any]:
    prior = ORIGINAL_VALIDATE()
    manifest = json.loads((FAILED_L49_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = FAILED_L49_ROOT / row["path"]
        actual = _sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expectedSha256": row["sha256"],
                "actualSha256": actual,
                "unchanged": actual == row["sha256"],
            }
        )
    passed = bool(prior["unchanged"] and rows and all(row["unchanged"] for row in rows))
    combined = [prior["aggregateSha256"], manifest["aggregateSha256"], *[row["actualSha256"] for row in rows]]
    return {
        **prior,
        "schema": "eidosoma.e01.s19_l49r.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "failedL49Unchanged": bool(rows and all(row["unchanged"] for row in rows)),
        "validatedFailedL49ArtifactCount": len(rows),
        "failedL49AggregateSha256": manifest["aggregateSha256"],
        "aggregateSha256": hashlib.sha256("|".join(combined).encode()).hexdigest(),
        "l49Rows": rows,
    }


def _availability_registry() -> pd.DataFrame:
    manifest = pd.read_parquet(L49.L23_ROOT / "input_trajectory_manifest.parquet")
    eligible = manifest[
        manifest["terminalStatus"].eq("requested_fissions_completed")
        & manifest["completedFissions"].ge(100)
        & manifest["selectedClockLength"].gt(max(L49.LANDMARKS) + 1)
    ]
    rows: list[dict[str, Any]] = []
    for source in eligible.itertuples(index=False):
        trajectory = L49.L28.load_trajectory(source)
        selected = tuple(
            L49.L28.selected_clock_observations(trajectory, L49.L28.CLOCK_ID)
        )
        counts = {
            landmark: sum(
                item.observation_kind == "post_fission"
                for item in selected[landmark:]
            )
            for landmark in L49.LANDMARKS
        }
        rows.append(
            {
                "candidateId": source.candidateId,
                "matrixIndex": int(source.matrixIndex),
                "selectedClockLength": len(selected),
                **{
                    f"futureFissionsAfterLandmark{landmark}": count
                    for landmark, count in counts.items()
                },
                "minimumFutureFissions": min(counts.values()),
                "availabilityEligible": min(counts.values()) >= L49.FISSION_HORIZON,
                "availabilityUsesScientificOutcome": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)


def select_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    firewall = pd.read_parquet(L49.L24_ROOT / "matrix_firewall.parquet")
    availability = _availability_registry()
    candidate_availability = availability.groupby("matrixIndex").agg(
        candidateCount=("candidateId", "nunique"),
        allCandidatesEligible=("availabilityEligible", "all"),
        minimumFutureFissions=("minimumFutureFissions", "min"),
    )
    shared = candidate_availability[
        candidate_availability["candidateCount"].eq(len(L49.CANDIDATES))
        & candidate_availability["allCandidatesEligible"]
    ].index
    rows: list[dict[str, Any]] = []
    failed_selection = pd.read_parquet(FAILED_L49_ROOT / "matrix_selection_registry.parquet")
    failed_selected = set(failed_selection["matrixIndex"].astype(int))
    for role in L49.ROLES:
        pool = firewall[
            firewall["matrixRole"].eq(role) & firewall["matrixIndex"].isin(shared)
        ].copy()
        pool["selectionDigest"] = pool["matrixIndex"].map(
            lambda matrix, role=role: hashlib.sha256(
                f"{FAILED_L49_VERSION}|STATE_SELECTION|{role}|{int(matrix)}".encode()
            ).hexdigest()
        )
        pool = pool.sort_values(["selectionDigest", "matrixIndex"])
        if len(pool) < L49.MATRICES_PER_ROLE:
            raise RuntimeError("insufficient availability-eligible shared matrices")
        for rank, row in enumerate(pool.head(L49.MATRICES_PER_ROLE).itertuples(), start=1):
            matrix = int(row.matrixIndex)
            rows.append(
                {
                    "matrixRole": role,
                    "matrixIndex": matrix,
                    "selectionRank": rank,
                    "selectionDigest": row.selectionDigest,
                    "eligibleSharedPool": len(pool),
                    "minimumFutureFissions": int(candidate_availability.loc[matrix, "minimumFutureFissions"]),
                    "availabilityEligible": True,
                    "retainedFromFailedL49": matrix in failed_selected,
                    "selectedBeforeBranchOutcome": True,
                }
            )
    selected = pd.DataFrame(rows).sort_values(["matrixRole", "selectionRank"]).reset_index(drop=True)
    expanded = pd.DataFrame(
        [
            {
                **row._asdict(),
                "candidateId": candidate,
                "landmark": landmark,
            }
            for row in selected.itertuples(index=False)
            for candidate in L49.CANDIDATES
            for landmark in L49.LANDMARKS
        ]
    ).sort_values(["matrixRole", "candidateId", "matrixIndex", "landmark"]).reset_index(drop=True)
    expanded["stateId"] = expanded.apply(
        lambda row: hashlib.sha256(
            f"{VERSION}|{row.matrixRole}|{row.candidateId}|{int(row.matrixIndex)}|{int(row.landmark)}".encode()
        ).hexdigest()[:24],
        axis=1,
    )
    if len(selected) != 40 or len(expanded) != 400 or selected["minimumFutureFissions"].min() < L49.FISSION_HORIZON:
        raise RuntimeError("L49R repaired availability scope failure")
    return selected, expanded


def seed_firewall(branches: pd.DataFrame, analysis: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in L49.ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if LOOP_ROOT in path.parents:
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            lower = column.lower()
            if "seedmaterialsha256" in lower:
                prior_material.update(frame[column].dropna().astype(str))
            if lower == "derivedseed" or lower.endswith("derivedseed"):
                prior_derived.update(frame[column].dropna().astype(str))
    current_material = set(analysis["seedMaterialSha256"].astype(str))
    current_derived = set(analysis["derivedSeed"].astype(str))
    for column in branches.columns:
        lower = column.lower()
        if "seedmaterialsha256" in lower:
            current_material.update(branches[column].dropna().astype(str))
        if lower.endswith("derivedseed"):
            current_derived.update(branches[column].dropna().astype(str))
    overlap_m = sorted(current_material & prior_material)
    overlap_d = sorted(current_derived & prior_derived)
    passed = not overlap_m and not overlap_d
    return {
        "schema": "eidosoma.e01.s19_l49r.seed_firewall.v1",
        "status": "PASS" if passed else "FAIL",
        "newBranchStreams": len(branches),
        "analysisStreams": len(analysis),
        "seedMaterialUnique": len(current_material) == len(branches) * 4 + len(analysis),
        "seedMaterialOverlapCount": len(overlap_m),
        "derivedSeedOverlapCount": len(overlap_d),
        "seedMaterialOverlaps": overlap_m,
        "derivedSeedOverlaps": overlap_d,
        "failedL49SeedsIncludedAsPrior": True,
    }


def source_registry() -> pd.DataFrame:
    frame = ORIGINAL_SOURCE_REGISTRY().copy()
    return pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "sourceId": "L49_PREOUTCOME_AVAILABILITY_FAILURE",
                        "evidenceClass": "DIRECT_FROZEN_E01_FAILURE",
                        "finding": "One originally selected state had only nine remaining fissions for the frozen F12 horizon; no scientific branch was executed.",
                        "frozenUse": "add future-fission availability to matrix eligibility and retain the original hash ranking",
                        "url": None,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )


def report_text(*args: Any, **kwargs: Any) -> str:
    text = ORIGINAL_REPORT(*args, **kwargs)
    text = text.replace(
        "# S19-L49 Full Results", "# S19-L49R Full Results"
    ).replace(" L49 ", " L49R ").replace("L49 therefore", "L49R therefore")
    repair = (
        "\n## Additive repair provenance\n\n"
        "Failed L49 remains immutable and released no scientific branch result. "
        "L49R changed only matrix eligibility: every selected shared matrix had to "
        "retain at least twelve post-fission boundaries after all five locked "
        "landmarks in both candidates. The original hash ranking, landmark set, "
        "F12 horizon, event, branch count, controls and gates were unchanged.\n"
    )
    return text + repair


def manifest_for(root: Path) -> dict[str, Any]:
    manifest = ORIGINAL_MANIFEST(root)
    manifest["schema"] = "eidosoma.e01.s19_l49r.artifact_manifest.v1"
    manifest["loopId"] = LOOP_ID
    return manifest


def append_ledgers(classifications: list[str], timestamp: str, next_theme: str) -> None:
    ledger_path = L49.ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "The L49 lock assumed selected-clock length implied sufficient F12 future-fission support.",
            "failureOrAmbiguityTargeted": "Pre-branch state availability.",
            "informationGainRationale": "Fail closed before simulation when a locked state cannot support its horizon.",
            "learned": "LOOP_FAILED_CLOSED;PREOUTCOME_STATE_AVAILABILITY_DESIGN_ERROR;NO_SCIENTIFIC_OUTCOME",
            "ledgerSequence": sequence,
            "loopId": "S19-L49",
            "motivatingEvidence": "State-restoration cardinality failure before branch execution.",
            "proposedNextTest": "S19-L49R outcome-blind availability repair.",
            "recordPhase": "POST_LOOP_PREOUTCOME_FAILURE",
            "remainingPlausibleHypotheses": "The unchanged longitudinal committor question remains untested.",
            "selectedHypotheses": "None; no scientific outcome opened.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Selected-clock length alone is a sufficient F12 availability check.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L49 failed solely because one matrix lacked the required future-fission support at the last landmark.",
            "failureOrAmbiguityTargeted": "Outcome-blind availability eligibility while preserving the scientific contract.",
            "informationGainRationale": "Replacing only ineligible matrices by the next original-ranked matrix adjudicates the unchanged question without moving landmarks or horizon.",
            "learned": "L49R availability repair locked before branch or realized-process outcome access.",
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Frozen L49 failure ledger and complete availability audit.",
            "proposedNextTest": "Execute the unchanged longitudinal process-risk analysis.",
            "recordPhase": "PRE_LOOP_REPAIR_LOCK",
            "remainingPlausibleHypotheses": "Within-lineage risk, stationary propensity, direct-history sufficiency or simulation-only forecasting.",
            "selectedHypotheses": "Longitudinal process committor after one availability-only repair.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "None scientifically; L49 released no outcome.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A useful risk trajectory must pass independent-half and independent-realized-future gates in both candidates.",
            "failureOrAmbiguityTargeted": "State dependence and independent-event calibration.",
            "informationGainRationale": "Whole-matrix inference respects five correlated states per catalytic matrix.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 2,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L49R result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Longitudinal online process-committor measurement.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any registered L49R measurement or forecast gate that failed.",
        },
    ]
    L49.BASE.write_parquet(
        ledger_path,
        pd.concat([ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)], ignore_index=True),
    )
    markdown = L49.ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    L49.BASE.atomic_text(
        markdown,
        markdown.read_text()
        + "\n\n## S19-L49 — preoutcome availability failure\n\n"
        + "- **Learned:** `LOOP_FAILED_CLOSED`; no scientific branch ran.\n"
        + "\n## S19-L49R — availability-only repair\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )
    candidate_path = L49.ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": L49.BRANCHES,
        "bundleId": "L49R_LONGITUDINAL_PROCESS_COMMITTOR",
        "candidateId": "S19-L49R-LONGITUDINAL-PROCESS-COMMITTOR",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 3,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 1,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 5,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "five fixed longitudinal states, 64 F12 shoots and an availability-only shared-matrix eligibility repair",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": "SIMULATION_ACCESSIBLE_PROCESS_PRECURSOR_LEAD" in classifications,
        "selectionReason": "L49_PREOUTCOME_AVAILABILITY_FAILURE",
        "sourceGrounding": 4,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    L49.BASE.write_parquet(
        candidate_path,
        pd.concat([candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)], ignore_index=True),
    )
    source_path = L49.ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_additions = []
    for row in source_registry().itertuples(index=False):
        source_additions.append(
            {
                "commitOrVersion": None,
                "evidenceClass": row.evidenceClass,
                "finding": f"{row.finding}; L49R use: {row.frozenUse}",
                "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L49R_{row.sourceId}",
                "sourceType": row.evidenceClass,
                "treeIdentity": None,
                "url": row.url,
            }
        )
    L49.BASE.write_parquet(
        source_path,
        pd.concat([sources, pd.DataFrame(source_additions).reindex(columns=sources.columns)], ignore_index=True),
    )


def _configure() -> None:
    L49.LOOP_ID = LOOP_ID
    L49.VERSION = VERSION
    L49.LOOP_ROOT = LOOP_ROOT
    L49.BUILD_ROOT = BUILD_ROOT
    L49.CONFIG = CONFIG
    L49.RUNNER_PATH = RUNNER_PATH
    L49.SEED_ROOT = SEED_ROOT
    L49.validate_immutable_prior = validate_immutable_prior
    L49.select_matrices = select_matrices
    L49.seed_firewall = seed_firewall
    L49.source_registry = source_registry
    L49.report_text = report_text
    L49.manifest_for = manifest_for
    L49.append_ledgers = append_ledgers


def prepare_lock() -> None:
    _configure()
    ORIGINAL_PREPARE()
    decision = (
        "# S19-L49R decision record\n\n"
        "Failed L49 is preserved unchanged and produced zero scientific branches. "
        "This one-repair-only additive step preserves all scientific settings and "
        "adds the logically required pre-outcome eligibility condition that each "
        "selected shared matrix retain at least twelve post-fission boundaries "
        "after every locked landmark in both candidates. Any ineligible matrix is "
        "replaced only by the next matrix in L49's already frozen SHA-256 ranking. "
        "No landmark, horizon, event, threshold, branch count, control, statistic "
        "or gate changes. A fresh seed root and cache are mandatory.\n"
    )
    L49.BASE.atomic_text(LOOP_ROOT / "decision_record.md", decision)
    implementation = json.loads((LOOP_ROOT / "implementation_lock.json").read_text())
    implementation.update(
        {
            "schema": "eidosoma.e01.s19_l49r.implementation_lock.v1",
            "repairOf": "S19-L49",
            "repairOnly": "MATRIX_F12_AVAILABILITY_ELIGIBILITY",
            "failedL49AggregateSha256": json.loads((FAILED_L49_ROOT / "artifact_manifest.json").read_text())["aggregateSha256"],
        }
    )
    L49.BASE.write_json(LOOP_ROOT / "implementation_lock.json", implementation)


def execute() -> None:
    _configure()
    ORIGINAL_EXECUTE()
    for name in (
        "classification.json",
        "runtime_manifest.json",
        "regeneration_validation.json",
        "storage_validation.json",
    ):
        path = LOOP_ROOT / name
        payload = json.loads(path.read_text())
        if isinstance(payload.get("schema"), str):
            payload["schema"] = payload["schema"].replace("s19_l49.", "s19_l49r.")
        payload["repairOf"] = "S19-L49"
        L49.BASE.write_json(path, payload)
    summary = (LOOP_ROOT / "loop_decision_summary.md").read_text().replace(
        "# S19-L49 decision summary", "# S19-L49R decision summary"
    )
    L49.BASE.atomic_text(LOOP_ROOT / "loop_decision_summary.md", summary)
    L49.BASE.write_json(LOOP_ROOT / "artifact_manifest.json", manifest_for(LOOP_ROOT))
    L49.BASE.write_json(L49.ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(L49.ARTIFACT_ROOT))


def main() -> None:
    if "--prepare-lock" in sys.argv:
        prepare_lock()
    else:
        execute()


if __name__ == "__main__":
    main()
