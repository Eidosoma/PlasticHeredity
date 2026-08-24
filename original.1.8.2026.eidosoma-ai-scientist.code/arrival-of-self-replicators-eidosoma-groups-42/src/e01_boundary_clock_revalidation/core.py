"""Frozen S12H contracts for candidate-1 C1 revalidation.

S12H never changes or reruns GARD dynamics.  It changes only the analysis clock
of the 32 locked candidate-1 raw trajectories from C0 to uniformly
boundary-inclusive C1 and evaluates the unchanged S12FR confirmation gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from e01_latent_timebase.inference import Particle, particle_summary_and_distance

VERSION = "E01-S12H-CANDIDATE1-BOUNDARY-CLOCK-REVALIDATION-v1.0.0"
RESEARCH_STEP_ID = "S12H"
EVIDENCE_CLASS = "SOURCE_INFORMED_BOUNDARY_CLOCK_REVALIDATED_ENSEMBLE_RECONSTRUCTION"
DERIVED_CANDIDATE_ID = "S12H-CANDIDATE-01-C1-DERIVATIVE-v1.0.0"
CANDIDATE_IDS = (
    "S12F-CANDIDATE-01",
    "S12F-CANDIDATE-02",
    "S12F-CANDIDATE-03",
)


def stage2_candidate_registry() -> list[dict[str, Any]]:
    """Return the exact three-candidate identities frozen for conditional stage 2."""

    return [
        {
            "candidateId": "S12F-CANDIDATE-01",
            "analysisIdentity": DERIVED_CANDIDATE_ID,
            "baseCandidateId": "S12F-CANDIDATE-01",
            "h": 0.5081160391061118,
            "daughterRule": "FIRST_DAUGHTER",
            "overshootRule": "RETAIN_OVERSHOOT",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "relationshipToS12FR": "NEW_BOUNDARY_INCLUSIVE_DERIVATIVE_NOT_ORIGINAL_C0",
        },
        {
            "candidateId": "S12F-CANDIDATE-02",
            "analysisIdentity": "S12F-CANDIDATE-02",
            "baseCandidateId": "S12F-CANDIDATE-02",
            "h": 0.6031526490073492,
            "daughterRule": "FIRST_DAUGHTER",
            "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "relationshipToS12FR": "UNCHANGED_CONFIRMED_CANDIDATE",
        },
        {
            "candidateId": "S12F-CANDIDATE-03",
            "analysisIdentity": "S12F-CANDIDATE-03",
            "baseCandidateId": "S12F-CANDIDATE-03",
            "h": 0.5613315384859516,
            "daughterRule": "RANDOM_NONEMPTY",
            "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "relationshipToS12FR": "UNCHANGED_CONFIRMED_CANDIDATE",
        },
    ]


def candidate1_revalidation_result(
    candidate: dict[str, Any],
    rows: pd.DataFrame,
    *,
    state_cardinality_passed: bool,
    seed_provenance_passed: bool,
    runtime_storage_passed: bool,
) -> dict[str, Any]:
    """Apply the original S12FR confirmation envelope to candidate 1 under C1."""

    if candidate["candidateId"] != "S12F-CANDIDATE-01":
        raise ValueError("S12H stage one may revalidate only S12F-CANDIDATE-01")
    if candidate["clockId"] != "C1_SELECTED_DAUGHTER_RETAINED":
        raise ValueError("S12H candidate 1 must use uniform C1")
    if len(rows) != 32:
        raise ValueError("S12H candidate-1 confirmation requires exactly 32 rows")

    values = rows["clockC1"].to_numpy(dtype=np.float64)
    q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
    particle = Particle(
        particle_id=str(candidate["representativeParticleId"]),
        family=str(candidate["exposureFamily"]),
        round_index=3,
        daughter_rule=str(candidate["daughterRule"]),
        overshoot_rule=str(candidate["overshootRule"]),
        clock_id="C1_SELECTED_DAUGHTER_RETAINED",
        h=float(candidate["h"]),
        c=None,
        h_max=None,
        parent_particle_id=None,
        proposal_weight=float(candidate["posteriorMass"]),
    )
    distance = particle_summary_and_distance(particle, rows)
    completed = int((rows["completedFissions"] == 100).sum())
    generation_denominator = max(1, int(rows["completedFissions"].sum()))
    maxsteps_fraction = float(rows["maxstepsTerminations"].sum() / generation_denominator)
    endpoints_inside = int(sum(q05 <= value <= q95 for value in (800.0, 800.0, 1000.0)))
    aggregate_compatible = bool(
        float(values.max()) >= 1090.0
        and q95 <= 1314.0
        and float(np.mean(values > 1314.0)) <= 0.05
    )
    median_post = float(rows["medianPostFissionMass"].median())
    replay_passed = bool(
        rows["repairedReplayPassed"].astype(bool).all()
        and rows["clockReplayPassed"].astype(bool).all()
        and int(rows["discreteDivergenceCount"].sum()) == 0
        and int(rows["finiteNumericDivergenceCount"].sum()) == 0
        and int(rows["forbiddenNonfiniteDifferenceCount"].sum()) == 0
        and int(rows["seedDifferenceCount"].sum()) == 0
    )
    cache_hash_passed = bool(rows["cacheHashPassed"].astype(bool).all())
    identity_passed = bool(rows["candidateIdentityPassed"].astype(bool).all())
    all_gates = {
        "exactly32LockedTrajectories": len(rows) == 32,
        "completedAtLeast31Of32": completed >= 31,
        # This is deliberately the original S12F/S12FR confirmation rule.
        # Although all three original candidates happened to cover all three
        # visible endpoints, the frozen gate required at least two of three.
        "atLeastTwoSampleEndpointsInsideQ05Q95": endpoints_inside >= 2,
        "aggregateSupportCompatible": aggregate_compatible,
        "medianPostFissionMassInside35To45": 35.0 <= median_post <= 45.0,
        "maxstepsFractionAtMost0p05": maxsteps_fraction <= 0.05,
        "confirmationDistanceAtMost1": float(distance["distance"]) <= 1.0,
        "uniformC1NoSyntheticDuplicate": True,
        "stateCardinality": state_cardinality_passed,
        "exactReplay": replay_passed,
        "cacheHashes": cache_hash_passed,
        "candidateIdentity": identity_passed,
        "seedAndProvenance": seed_provenance_passed,
        "runtimeAndStorage": runtime_storage_passed,
    }
    reasons = [name for name, passed in all_gates.items() if not passed]
    return {
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "candidateId": candidate["candidateId"],
        "derivedCandidateId": DERIVED_CANDIDATE_ID,
        "baseClockId": "C0_BATCH_UPDATES_ONLY",
        "revalidatedClockId": "C1_SELECTED_DAUGHTER_RETAINED",
        "clockPolicy": "selected_daughter_recorded_after_every_fission_without_exception",
        "exposureFamily": candidate["exposureFamily"],
        "h": candidate["h"],
        "daughterRule": candidate["daughterRule"],
        "overshootRule": candidate["overshootRule"],
        "completedLineages": completed,
        "q05TPhi": float(q05),
        "medianTPhi": float(q50),
        "q95TPhi": float(q95),
        "maximumTPhi": float(values.max()),
        "fractionBeyondAxisUpper1314": float(np.mean(values > 1314.0)),
        "sampleEndpointsInsideQ05Q95": endpoints_inside,
        "aggregateCompatible": aggregate_compatible,
        "medianPostFissionMass": median_post,
        "q95Overshoot": float(rows["q95Overshoot"].quantile(0.95)),
        "fractionMaxsteps": maxsteps_fraction,
        "confirmationDistance": float(distance["distance"]),
        "distanceComponents": {
            "D_T": float(distance["D_T"]),
            "D_M": float(distance["D_M"]),
            "D_O": float(distance["D_O"]),
            "complexity": float(distance["complexity"]),
        },
        "gates": all_gates,
        "confirmationGatePassed": all(all_gates.values()),
        "gateReason": "PASS" if not reasons else ";".join(reasons),
    }
