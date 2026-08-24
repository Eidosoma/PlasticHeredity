from __future__ import annotations

from plastic_heredity import intervention_cr6_transfer as cr6
from plastic_heredity.intervention_cr6_readback_amendment import (
    FAILED_REGIME,
    ORIGINAL_REGISTRATION,
    ORIGINAL_REGISTRATION_ID,
    _artificial_round_trip,
    corrected_readback_regime,
)


def test_amendment_preserves_original_scientific_registration() -> None:
    registration = cr6.verify_registration(ORIGINAL_REGISTRATION)
    assert registration["registration_id"] == ORIGINAL_REGISTRATION_ID
    assert registration["source_hashes"] == cr6.source_hashes()


def test_original_false_failure_and_corrected_exact_readback() -> None:
    audit = _artificial_round_trip()
    assert audit["original_false_failure_reproduced"] is True
    assert audit["amended_metrics_exact"] is True
    assert audit["amended_matrix_rows_exact"] is True
    assert audit["only_added_field"] == "regime"
    assert audit["numerical_tolerance_weakened"] is False


def test_amendment_is_a_callback_overlay_not_a_scientific_runner() -> None:
    assert FAILED_REGIME in cr6.REGIMES
    assert corrected_readback_regime is not cr6._readback_regime
    assert cr6.phase_spec(FAILED_REGIME).arms == (
        "MODEL_UP",
        "MODEL_DOWN",
        "RANDOM",
        "NOOP",
    )
    assert cr6.protocol()["operational"]["cr7_not_launched_automatically"] is True
