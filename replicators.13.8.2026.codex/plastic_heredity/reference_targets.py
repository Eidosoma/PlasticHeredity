from __future__ import annotations

from typing import Any


# Ranges transcribed from PRE_PRINT_PAPER_DRAFT.md and its embedded Distill plots.
# They are comparison targets only; the simulator and model never read them.
REPORTED_TARGETS: dict[str, Any] = {
    "process": {
        "break_event": [0.64, 0.73],
        "resume_2": [0.88, 0.91],
        "episode_3": [0.76, 0.82],
        "persist_5": [0.53, 0.60],
        "old_return": [0.0024, 0.0069],
        "old_anchor_gain_mean": [-0.28, -0.24],
    },
    "confirmation": {
        "branch_half_spearman": [0.924, 0.938],
        "centered_branch_half_spearman": [0.606, 0.625],
        "full_overall_spearman": [0.895, 0.918],
        "full_centered_spearman": [0.550, 0.697],
        "history_overall_spearman": [0.742, 0.822],
        "history_centered_spearman": [0.198, 0.345],
        "beta_overall_abs_max": 0.10,
        "log_loss_gain": [0.041, 0.052],
        "q_brier_gain": [0.012, 0.018],
        # Exact minimum for the registered (exceedances + 1)/(512 + 1) test;
        # the supplied manuscript rounds this to 0.001949.
        "permutation_p_max": 1.0 / 513.0,
    },
}


def in_range(value: float, bounds: list[float]) -> bool:
    return bounds[0] <= value <= bounds[1]
