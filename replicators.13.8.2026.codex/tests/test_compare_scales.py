from scripts.compare_scales import _candidate_metrics


def test_candidate_metrics_computes_predeclared_model_advantage():
    metrics = {
        "02": {
            "branch_half_reliability": 0.9,
            "branch_half_reliability_lower_95": 0.8,
            "centered_branch_half_reliability": 0.7,
            "centered_branch_half_reliability_lower_95": 0.6,
            "transition_region_states": 8,
            "states": 10,
            "models": {
                "full": {
                    "overall_spearman_mean": 0.8,
                    "centered_spearman_mean": 0.5,
                },
                "history": {
                    "overall_spearman_mean": 0.6,
                    "centered_spearman_mean": 0.2,
                },
            },
            "directions": {
                "A": {
                    "log_loss_gain": 0.04,
                    "log_loss_gain_ci95": [0.01, 0.07],
                    "q_brier_gain": 0.02,
                    "q_brier_gain_ci95": [0.005, 0.03],
                },
                "B": {
                    "log_loss_gain": 0.02,
                    "log_loss_gain_ci95": [0.002, 0.05],
                    "q_brier_gain": 0.01,
                    "q_brier_gain_ci95": [0.003, 0.02],
                },
            },
        }
    }
    values = _candidate_metrics(metrics, "02")
    assert values["full_minus_history_centered"] == 0.3
    assert values["mean_log_loss_gain"] == 0.03
    assert values["minimum_log_loss_gain_lower_95"] == 0.002
    assert values["minimum_q_brier_gain_lower_95"] == 0.003
    assert values["transition_region_fraction"] == 0.8
