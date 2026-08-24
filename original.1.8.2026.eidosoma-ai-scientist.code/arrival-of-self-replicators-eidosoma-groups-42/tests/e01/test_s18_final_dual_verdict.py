from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/e01/run_s18_final_dual_verdict.py"
SPEC = importlib.util.spec_from_file_location("s18_final_dual_verdict", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
S18 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S18)


def test_frozen_claim_adjudication_covers_exact_ledger() -> None:
    contract = S18.load_contract()
    ledger = S18.read_csv(S18.LEGACY_BUNDLE_ROOT / "ledgers/claim_ledger.csv")
    mapping = S18.load_claim_map()
    checks = S18.validate_contract(contract, ledger, mapping)
    assert all(checks.values())
    assert len(mapping) == 59
    assert set(mapping) == {row["claim_id"] for row in ledger}


def test_status_counts_and_directional_exact_separation_are_frozen() -> None:
    contract = S18.load_contract()
    mapping = S18.load_claim_map()
    assert dict(Counter(row["status"] for row in mapping.values())) == contract[
        "expectedClaimStatusCounts"
    ]
    assert contract["directionalPolicy"]["exactNumbersAreNotRequiredForDirectionalSupport"]
    assert contract["directionalPolicy"]["exactAndDirectionalAssessmentsAreSeparate"]
    assert contract["directionalPolicy"]["favorableCandidateSelectionForbidden"]
    assert mapping["E01-C026"]["directional_assessment"] == "MIXED_ACROSS_REQUIRED_BRANCHES"
    assert mapping["E01-C026"]["status"] == "NOT_SUPPORTED_WITHIN_TESTED_SCOPE"


def test_matrix_b_and_figure_table_map_are_complete_and_separate() -> None:
    contract = S18.load_contract()
    assert len(contract["matrixB"]) == 7
    assert {row["questionId"] for row in contract["matrixB"]} == {
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
    }
    assert {row["componentId"] for row in contract["figureTableMap"]} == {
        "FIGURE_2",
        "FIGURE_3",
        "FIGURE_4",
        "FIGURE_5",
        "FIGURE_6",
        "TABLE_1",
    }
    assert next(row for row in contract["matrixB"] if row["questionId"] == "B07")[
        "status"
    ] == "NOT_SUPPORTED_WITHIN_TESTED_SCOPE"


def test_required_final_classifications_and_continuation_options() -> None:
    contract = S18.load_contract()
    classifications = {
        row["classification"] for row in contract["finalClassifications"]
    }
    assert S18.REQUIRED_CLASSIFICATIONS <= classifications
    options = {row["optionId"]: row for row in contract["postCloseoutHumanReviewOptions"]}
    assert options["OPTION_A_STRONGER_E02"]["recommendationRank"] == 1
    assert options["OPTION_B_VERSIONED_E01_REOPEN_S19_S20"]["authorizationRequired"]
    assert not options["OPTION_B_VERSIONED_E01_REOPEN_S19_S20"]["active"]
    assert contract["currentBoundaryAfterS18"] == {
        "e01Closed": True,
        "s19S20Queued": False,
        "e02Started": False,
        "chiefAndHumanMayAuthorizeFutureVersionedContinuation": True,
    }


def test_claim_component_and_dependency_boundaries() -> None:
    assert S18.claim_component(13) == "FIGURE_2"
    assert S18.claim_component(18) == "FIGURE_3"
    assert S18.claim_component(21) == "FIGURE_4"
    assert S18.claim_component(28) == "FIGURE_5"
    assert S18.claim_component(40) == "TABLE_1"
    assert S18.claim_component(50) == "FIGURE_6"
    assert S18.claim_dependencies(18) == (
        "YES_PRIMARY_PAPER_FACING_VALUES_ARE_COMPLETED_FIT",
        "YES_Y_EQUALS_I_H_GT_0_9",
        "NO",
    )
    assert S18.claim_dependencies(46)[2] == "YES_APPEND_AND_REFIT_CURRENT_PREFIX"
