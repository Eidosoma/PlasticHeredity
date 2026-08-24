from __future__ import annotations

import pandas as pd

from e01_boundary_clock_revalidation import core


def _candidate() -> dict[str, object]:
    return {
        "candidateId": "S12F-CANDIDATE-01",
        "representativeParticleId": "FIXED_COMMON_EXPOSURE-R3-P030",
        "exposureFamily": "FIXED_COMMON_EXPOSURE",
        "h": 0.5081160391061118,
        "daughterRule": "FIRST_DAUGHTER",
        "overshootRule": "RETAIN_OVERSHOOT",
        "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
        "posteriorMass": 0.045291437504119944,
    }


def _rows() -> pd.DataFrame:
    # A compact gate fixture with two visible endpoints in q05--q95 and an
    # aggregate-supporting maximum. The scientific distance is patched below
    # so this unit test isolates S12H's unchanged confirmation predicates.
    values = [800.0] * 31 + [1100.0]
    return pd.DataFrame(
        {
            "clockC1": values,
            "completedFissions": [100] * 32,
            "maxstepsTerminations": [0] * 32,
            "medianPostFissionMass": [40.0] * 32,
            "q95Overshoot": [5.0] * 32,
            "repairedReplayPassed": [True] * 32,
            "clockReplayPassed": [True] * 32,
            "discreteDivergenceCount": [0] * 32,
            "finiteNumericDivergenceCount": [0] * 32,
            "forbiddenNonfiniteDifferenceCount": [0] * 32,
            "seedDifferenceCount": [0] * 32,
            "cacheHashPassed": [True] * 32,
            "candidateIdentityPassed": [True] * 32,
        }
    )


def test_registry_changes_only_candidate_one_clock_identity() -> None:
    registry = core.stage2_candidate_registry()
    assert len(registry) == 3
    assert registry[0]["analysisIdentity"] == core.DERIVED_CANDIDATE_ID
    assert registry[0]["clockId"] == "C1_SELECTED_DAUGHTER_RETAINED"
    assert registry[0]["relationshipToS12FR"] == (
        "NEW_BOUNDARY_INCLUSIVE_DERIVATIVE_NOT_ORIGINAL_C0"
    )
    assert all(item["clockId"] == "C1_SELECTED_DAUGHTER_RETAINED" for item in registry)
    assert registry[1]["relationshipToS12FR"] == "UNCHANGED_CONFIRMED_CANDIDATE"
    assert registry[2]["relationshipToS12FR"] == "UNCHANGED_CONFIRMED_CANDIDATE"


def test_original_confirmation_gate_requires_two_not_three_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        core,
        "particle_summary_and_distance",
        lambda particle, rows: {
            "distance": 0.5,
            "D_T": 0.1,
            "D_M": 0.0,
            "D_O": 0.0,
            "complexity": 1.0,
        },
    )
    result = core.candidate1_revalidation_result(
        _candidate(),
        _rows(),
        state_cardinality_passed=True,
        seed_provenance_passed=True,
        runtime_storage_passed=True,
    )
    assert result["sampleEndpointsInsideQ05Q95"] == 2
    assert result["gates"]["atLeastTwoSampleEndpointsInsideQ05Q95"] is True
    assert result["confirmationGatePassed"] is True


def test_any_replay_failure_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        core,
        "particle_summary_and_distance",
        lambda particle, rows: {
            "distance": 0.5,
            "D_T": 0.1,
            "D_M": 0.0,
            "D_O": 0.0,
            "complexity": 1.0,
        },
    )
    rows = _rows()
    rows.loc[0, "repairedReplayPassed"] = False
    result = core.candidate1_revalidation_result(
        _candidate(),
        rows,
        state_cardinality_passed=True,
        seed_provenance_passed=True,
        runtime_storage_passed=True,
    )
    assert result["gates"]["exactReplay"] is False
    assert result["confirmationGatePassed"] is False
    assert "exactReplay" in result["gateReason"]
