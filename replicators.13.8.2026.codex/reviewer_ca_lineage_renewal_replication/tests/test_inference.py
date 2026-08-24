from __future__ import annotations

from copy import deepcopy

from reviewer_ca_lineage_renewal_replication.contract import CONDITIONS
from reviewer_ca_lineage_renewal_replication.inference import adjudicate


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
    values = {condition: 0.0 for condition in CONDITIONS}
    values.update(
        {
            "intact": 0.5,
            "no_rewrite": 0.05,
            "rescue_same_enter_g4": 0.7,
            "rescue_opposite_enter_g4": -0.7,
            "opposite_founder": -0.5,
            "carrier_corruption_1": 0.45,
        }
    )
    conditions = {}
    for condition in CONDITIONS:
        outcomes = {}
        carrier_history = {}
        for generation in (1, 2, 4, 8, 16):
            crossover = values[condition]
            if condition == "intact":
                crossover = {1: 0.75, 2: 0.74, 4: 0.70, 8: 0.50, 16: 0.25}[generation]
            if condition == "ablate_after_g2" and generation < 4:
                crossover = 0.7
            if condition.startswith("rescue_") and generation < 4:
                crossover = 0.7 if generation < 3 else 0.0
            outcomes[str(generation)] = {
                "primary": _row(crossover),
                "terminal": _row(crossover),
                "survival": 1.0,
            }
            carrier_history[str(generation)] = {
                "entry": {"centroid_l2": 4.0, "mean_abs": 0.5 ** generation, "within_history_variance": 0.1},
                "exit": {"centroid_l2": 5.0, "mean_abs": 0.4, "within_history_variance": 0.1},
                "surviving_futures": 128,
            }
        conditions[condition] = {"outcomes": outcomes, "carrier_history": carrier_history}
    return {"pair_id": f"pair-{index}", "conditions": conditions}


def test_strict_synthetic_panel_passes_every_gate() -> None:
    payloads = [_payload(index) for index in range(8)]
    result = adjudicate(payloads, complete=True, expected_pairs=8, resamples=200)
    assert result["verdict"] == "STRICT_RENEWED_CA_PLASTIC_HEREDITY"
    assert result["strict_primary_passed"] is True
    assert all(result["gates"].values())


def test_incomplete_confirmation_cannot_pass() -> None:
    payloads = [_payload(index) for index in range(7)]
    result = adjudicate(payloads, complete=False, expected_pairs=8, resamples=100)
    assert result["verdict"] == "INCOMPLETE"


def test_static_founder_template_is_distinguished_from_active_rewrite() -> None:
    payloads = [_payload(index) for index in range(8)]
    for payload in payloads:
        payload["conditions"]["no_rewrite"]["outcomes"]["8"]["primary"] = _row(0.48)
    result = adjudicate(payloads, complete=True, expected_pairs=8, resamples=100)
    assert result["verdict"] == "STATIC_HIDDEN_TEMPLATE"
