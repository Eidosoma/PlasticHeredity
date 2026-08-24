from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from plastic_heredity import intervention_replication as base
from plastic_heredity.config import GardConfig
from plastic_heredity.intervention_cr6_transfer import (
    ARMS,
    BOOTSTRAP_REPETITIONS,
    BRANCHES,
    DEFAULT_CPU_BUDGET_HOURS,
    HORIZON,
    LANDMARKS,
    MATRICES,
    MAXIMUM_CPU_BUDGET_HOURS,
    MINIMUM_CPU_BUDGET_HOURS,
    NULL_EQUIVALENCE_MARGIN,
    NULL_REGIME,
    POSITIVE_REGIMES,
    RANDOMIZATION_REPETITIONS,
    RANDOM_EQUIVALENCE_MARGIN,
    REGIMES,
    SEEDS,
    _fixture_case,
    _fixture_inference,
    _prepare_work,
    _tost_equivalent,
    phase_spec,
    protocol,
    regime_gard,
)


def test_cr6_design_is_frozen_bounded_and_zero_shot() -> None:
    frozen = protocol()
    assert MATRICES == 40
    assert LANDMARKS == (35, 65)
    assert BRANCHES == 48
    assert HORIZON == 12
    assert ARMS == ("MODEL_UP", "MODEL_DOWN", "RANDOM", "NOOP")
    assert BOOTSTRAP_REPETITIONS == 4_096
    assert RANDOMIZATION_REPETITIONS == 4_096
    assert MINIMUM_CPU_BUDGET_HOURS == 3.0
    assert DEFAULT_CPU_BUDGET_HOURS == 5.0
    assert MAXIMUM_CPU_BUDGET_HOURS == 6.0
    assert frozen["frozen_model"]["zero_shot"] is True
    assert (
        frozen["frozen_model"]["refit_recalibration_search_or_regime_switching"]
        is False
    )
    assert frozen["operational"]["cr7_not_launched_automatically"] is True


def test_cr6_regimes_and_roles_are_exact() -> None:
    assert REGIMES == {
        "POS_A_M4_S5": (-4.0, 5.0, "positive_transfer"),
        "POS_A_M3_S4": (-3.0, 4.0, "positive_transfer"),
        "POS_A_M5_S4": (-5.0, 4.0, "positive_transfer"),
        "NULL_A_M4_S3": (-4.0, 3.0, "predicted_null"),
    }
    assert POSITIVE_REGIMES == (
        "POS_A_M4_S5",
        "POS_A_M3_S4",
        "POS_A_M5_S4",
    )
    assert NULL_REGIME == "NULL_A_M4_S3"


def test_only_beta_distribution_parameters_change_between_regimes() -> None:
    baseline = asdict(GardConfig())
    for regime, (a, sigma, _role) in REGIMES.items():
        observed = asdict(regime_gard(regime))
        changed = {key for key in observed if observed[key] != baseline[key]}
        expected = {
            key
            for key, value in (
                ("beta_log_mean", a),
                ("beta_log_sd", sigma),
            )
            if value != baseline[key]
        }
        assert changed == expected
        assert observed["beta_log_mean"] == a
        assert observed["beta_log_sd"] == sigma


def test_cr6_uses_original_model_guided_arms_with_fresh_arm_free_seeds() -> None:
    case = _fixture_case("02", 3, 35)
    assert len(SEEDS) == len(set(SEEDS.values()))
    assert set(SEEDS.values()).isdisjoint(base.SEED_DOMAINS.values())
    for regime in REGIMES:
        spec = phase_spec(regime)
        assert spec.phase == "p1"
        assert spec.arms == ARMS
        assert spec.contrast == ("MODEL_UP", "MODEL_DOWN")
        assert len({base._future_seed(spec, case, 7) for _arm in ARMS}) == 1
        assert base._selection_seed(spec, case, "random_edit") != base._future_seed(
            spec, case, 7
        )


def test_positive_fixture_passes_registered_four_cell_gate() -> None:
    metrics, rows = _fixture_inference(targeted_effect=True)
    assert metrics["positive_transfer_gate_pass"] is True
    assert metrics["predicted_null_gate_pass"] is None
    assert metrics["registered_regime_gate_pass"] is True
    assert len(metrics["cells"]) == 4
    assert all(cell["cr6_positive_transfer_cell_pass"] for cell in metrics["cells"])
    assert len(rows) == 6 * MATRICES


def test_null_fixture_uses_candidate_pooled_tost() -> None:
    metrics, rows = _fixture_inference(targeted_effect=False)
    assert metrics["positive_transfer_gate_pass"] is None
    assert metrics["predicted_null_gate_pass"] is True
    assert metrics["registered_regime_gate_pass"] is True
    assert len(metrics["candidate_pooled"]) == 2
    assert all(
        item["targeted_null_equivalence"]["tost_equivalent"]
        for item in metrics["candidate_pooled"]
    )
    assert len(rows) == 6 * MATRICES


def test_crossing_zero_alone_does_not_establish_equivalence() -> None:
    assert NULL_EQUIVALENCE_MARGIN == 0.04
    assert RANDOM_EQUIVALENCE_MARGIN == 0.025
    assert _tost_equivalent((-0.039, 0.039), 0.04)
    assert not _tost_equivalent((-0.05, 0.01), 0.04)
    assert not _tost_equivalent((-0.01, 0.05), 0.04)


def test_work_contract_enforces_cpu_bounds_and_is_resume_stable(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    output = tmp_path / "result"
    _prepare_work(work, output, "fixture-registration", 5.0)
    _prepare_work(work, output, "fixture-registration", 5.0)
    assert (work / "campaign_contract.json").is_file()
    for invalid in (2.99, 6.01):
        try:
            _prepare_work(tmp_path / f"invalid-{invalid}", output, "fixture", invalid)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("CR6 accepted a CPU declaration outside its bounds")


def test_cr6_claim_boundaries_exclude_strict_eight_and_broad_claims() -> None:
    frozen = protocol()
    prohibited = frozen["claim_boundary"]["prohibited"]
    assert frozen["endpoint"]["strict_eight_excluded"] is True
    assert "strict-eight control" in prohibited
    assert "biological memory" in prohibited
    assert "Phi or PhiID intervention" in prohibited
    assert frozen["selection"]["optional_rule_family_included"] is False
