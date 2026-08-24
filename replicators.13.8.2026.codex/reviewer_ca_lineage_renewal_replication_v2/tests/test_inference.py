from __future__ import annotations

from copy import deepcopy
import json

from reviewer_ca_lineage_renewal_replication_v2.contract import CONDITIONS
from reviewer_ca_lineage_renewal_replication_v2.inference import adjudicate, safe_fraction


def _row(crossover: float) -> dict[str, float]:
    upper = 0.5 + crossover / 2.0
    lower = 0.5 - crossover / 2.0
    return {
        "p_a_given_a": upper,
        "p_a_given_b": lower,
        "p_b_given_a": lower,
        "p_b_given_b": upper,
        "direction_a": crossover,
        "direction_b": crossover,
        "crossover": crossover,
        "correct": upper,
        "resolved": 1.0,
    }


def _payload(index: int) -> dict:
    condition_crossovers = {
        "intact": 0.50,
        "zero_every_boundary": 0.0,
        "shuffle_every_boundary": 0.0,
        "read_disabled": 0.0,
        "founder_write_disabled": 0.0,
        "no_rewrite": 0.05,
        "ablate_after_g2": 0.0,
        "rescue_same_enter_g4": 0.50,
        "rescue_opposite_enter_g4": -0.50,
        "opposite_founder": -0.50,
        "carrier_corruption_1": 0.40,
    }
    conditions = {}
    for name in CONDITIONS:
        outcomes = {
            str(generation): {
                "primary": _row(condition_crossovers[name]),
                "terminal": _row(condition_crossovers[name]),
                "survival": 1.0,
            }
            for generation in (1, 2, 4, 8, 16)
        }
        carrier_history = {
            str(generation): {
                "entry": {"centroid_l2": 1.0, "mean_abs": 0.5, "within_history_variance": 0.1},
                "exit": {"centroid_l2": 1.0, "mean_abs": 0.5, "within_history_variance": 0.1},
                "surviving_futures": 128,
            }
            for generation in (1, 2, 4, 8, 16)
        }
        conditions[name] = {"outcomes": outcomes, "carrier_history": carrier_history}
    panel = {
        "intact": 0.80,
        "no_rewrite": 0.55,
        "read_disabled": 0.50,
        "ablate_after_g2": 0.50,
        "rescue_same_enter_g4": 0.75,
        "rescue_opposite_enter_g4": 0.80,
    }
    return {
        "pair_id": f"pair-{index}",
        "conditions": conditions,
        "secondary_decoder": {
            "generation": 16,
            "carrier": dict(panel),
            "phenotype": dict(panel),
        },
    }


def test_complete_strict_and_secondary_verdict() -> None:
    payloads = [_payload(index) for index in range(8)]
    result = adjudicate(payloads, complete=True, expected_pairs=8, resamples=200)
    assert result["verdict"] == "STRICT_RENEWED_CA_PLASTIC_HEREDITY"
    assert result["strict_primary_passed"] is True
    assert all(result["strict_gates"].values())
    assert result["carrier_decoder"]["passed"] is True
    assert result["phenotype_decoder"]["passed"] is True


def test_incomplete_confirmation_cannot_pass() -> None:
    payloads = [_payload(index) for index in range(7)]
    result = adjudicate(payloads, complete=False, expected_pairs=8, resamples=100)
    assert result["verdict"] == "INCOMPLETE"


def test_drifted_and_cryptic_verdicts_are_available() -> None:
    payloads = [_payload(index) for index in range(8)]
    for payload in payloads:
        payload["conditions"]["intact"]["outcomes"]["16"]["primary"] = _row(0.0)
    result = adjudicate(payloads, complete=True, expected_pairs=8, resamples=100)
    assert result["verdict"] == "EXPRESSED_DRIFTED_LINEAGE_HEREDITY"
    for payload in payloads:
        payload["secondary_decoder"]["phenotype"]["no_rewrite"] = 0.78
    result = adjudicate(payloads, complete=True, expected_pairs=8, resamples=100)
    assert result["verdict"] == "CRYPTIC_RENEWED_CARRIER_MEMORY"


def test_undefined_ratio_is_structured_and_json_safe() -> None:
    ratio = safe_fraction(1.0, 0.0)
    assert ratio == {
        "defined": False,
        "value": None,
        "numerator": 1.0,
        "denominator": 0.0,
        "reason": "non_positive_denominator",
    }
    payloads = [_payload(index) for index in range(8)]
    for payload in payloads:
        payload["conditions"]["intact"]["outcomes"]["8"]["primary"] = _row(0.0)
    result = adjudicate(payloads, complete=True, expected_pairs=8, resamples=100)
    assert result["no_rewrite_loss_fraction"]["defined"] is False
    json.dumps(result, allow_nan=False)


def test_static_and_transient_failures_are_distinguished() -> None:
    payloads = [_payload(index) for index in range(8)]
    for payload in payloads:
        payload["conditions"]["no_rewrite"]["outcomes"]["8"]["primary"] = _row(0.48)
        for kind in ("carrier", "phenotype"):
            payload["secondary_decoder"][kind]["intact"] = 0.50
    result = adjudicate(payloads, complete=True, expected_pairs=8, resamples=100)
    assert result["verdict"] == "STATIC_HIDDEN_TEMPLATE"
    transient = [_payload(index) for index in range(8)]
    for payload in transient:
        payload["conditions"]["intact"]["outcomes"]["16"]["primary"] = _row(0.0)
        for kind in ("carrier", "phenotype"):
            payload["secondary_decoder"][kind]["intact"] = 0.50
    result = adjudicate(transient, complete=True, expected_pairs=8, resamples=100)
    assert result["verdict"] == "TRANSIENT_LINEAGE_MEMORY"


def test_no_durable_renewal_verdict_is_available() -> None:
    payloads = [_payload(index) for index in range(8)]
    for payload in payloads:
        payload["conditions"]["intact"]["outcomes"]["8"]["primary"] = _row(0.0)
        payload["conditions"]["intact"]["outcomes"]["16"]["primary"] = _row(0.0)
        for kind in ("carrier", "phenotype"):
            payload["secondary_decoder"][kind]["intact"] = 0.50
    result = adjudicate(payloads, complete=True, expected_pairs=8, resamples=100)
    assert result["verdict"] == "NO_DURABLE_RENEWAL"
