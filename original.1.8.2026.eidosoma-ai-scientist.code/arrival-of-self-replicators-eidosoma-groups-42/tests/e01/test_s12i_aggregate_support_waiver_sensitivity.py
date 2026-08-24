from __future__ import annotations

from e01_aggregate_support_waiver_sensitivity import core


def _s12h_confirmation() -> dict[str, object]:
    return {
        "confirmationGatePassed": False,
        "aggregateCompatible": False,
        "sampleEndpointsInsideQ05Q95": 3,
        "fractionBeyondAxisUpper1314": 0.0625,
        "gates": {
            "exactly32LockedTrajectories": True,
            "completedAtLeast31Of32": True,
            "atLeastTwoSampleEndpointsInsideQ05Q95": True,
            "aggregateSupportCompatible": False,
            "medianPostFissionMassInside35To45": True,
            "maxstepsFractionAtMost0p05": True,
            "confirmationDistanceAtMost1": True,
            "uniformC1NoSyntheticDuplicate": True,
            "stateCardinality": True,
            "exactReplay": True,
            "cacheHashes": True,
            "candidateIdentity": True,
            "seedAndProvenance": True,
            "runtimeAndStorage": True,
        },
    }


def _candidate_rows(
    *, prefix: tuple[bool, bool, bool], full: tuple[bool, bool, bool]
) -> list[dict[str, object]]:
    return [
        {
            "candidateId": candidate_id,
            "primaryPrefixGate": prefix[index],
            "primaryFullCoherent": full[index],
        }
        for index, candidate_id in enumerate(core.CANDIDATE_IDS)
    ]


def test_registry_marks_only_candidate_one_as_waived_and_nonconfirmed() -> None:
    registry = core.sensitivity_candidate_registry()
    assert len(registry) == 3
    assert all(item["clockId"] == "C1_SELECTED_DAUGHTER_RETAINED" for item in registry)
    assert registry[0]["analysisIdentity"] == core.DERIVED_CANDIDATE_ID
    assert registry[0]["aggregateSupportGateWaived"] is True
    assert registry[0]["upstreamConfirmed"] is False
    assert registry[0]["evidenceStatus"] == "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED"
    assert all(item["aggregateSupportGateWaived"] is False for item in registry[1:])
    assert all(item["upstreamConfirmed"] is True for item in registry[1:])


def test_exact_waiver_retains_the_original_failure() -> None:
    result = core.validate_exact_waiver(_s12h_confirmation())
    assert result["passed"] is True
    assert result["failedGateNames"] == ["aggregateSupportCompatible"]
    assert result["allNonwaivedGatesPassed"] is True
    assert result["waiverTreatedAsPass"] is False
    assert result["candidate1UpstreamConfirmed"] is False


def test_waiver_validation_rejects_any_second_failed_gate() -> None:
    confirmation = _s12h_confirmation()
    confirmation["gates"]["exactReplay"] = False  # type: ignore[index]
    result = core.validate_exact_waiver(confirmation)
    assert result["passed"] is False
    assert result["failedGateNames"] == ["aggregateSupportCompatible", "exactReplay"]


def test_all_prefix_positive_is_exploratory_only() -> None:
    result = core.sensitivity_classification(
        _candidate_rows(prefix=(True, True, True), full=(True, True, True))
    )
    assert result == "EXPLORATORY_SENSITIVITY_SET_PROSPECTIVE_POSITIVE_CONSISTENCY"


def test_all_full_positive_without_prefix_is_retrospective_exploratory() -> None:
    result = core.sensitivity_classification(
        _candidate_rows(prefix=(False, False, False), full=(True, True, True))
    )
    assert result == "EXPLORATORY_SENSITIVITY_SET_RETROSPECTIVE_POSITIVE_CONSISTENCY"


def test_any_primary_gate_disagreement_is_candidate_sensitive() -> None:
    result = core.sensitivity_classification(
        _candidate_rows(prefix=(False, False, False), full=(True, False, False))
    )
    assert result == "CANDIDATE_SENSITIVE_UNDERDETERMINED"


def test_no_primary_gate_pass_is_sensitivity_set_non_support() -> None:
    result = core.sensitivity_classification(
        _candidate_rows(prefix=(False, False, False), full=(False, False, False))
    )
    assert result == "SENSITIVITY_SET_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE"
