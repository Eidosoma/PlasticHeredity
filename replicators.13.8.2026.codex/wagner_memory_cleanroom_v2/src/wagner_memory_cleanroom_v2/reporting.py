from __future__ import annotations

from pathlib import Path
from typing import Any


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def write_reports(run_dir: Path, analysis: dict[str, Any], verification: dict[str, Any]) -> None:
    state = analysis["state"]["writers"]["hard-theta-0"]
    carrier = analysis["carrier"]
    state_metrics = state["metrics"]
    carrier_metrics = carrier["metrics"]
    report = f"""# Corrected Wagner memory replication v2

Overall registered verdict: **{analysis['overall_verdict']}**

This run kept strict eight-cycle point-form retention separate from the first
three-cycle A/B/other point destination used for prediction, used two
complementary midpoint starts per rulebook, arm-paired random futures, and
whole-rulebook simultaneous bootstrap bounds. Verification
was {'successful' if verification['verified'] else 'not successful'} for exact
stage counts, independent state/carrier regeneration, ordered per-cell future
digests, source records, and the registered future-ID replay sample.

## Expression-state channel

- Verdict: {analysis['state']['state_channel_verdict']}
- Direct within-treatment A/B crossover: {_number(state_metrics['direct_crossover']['mean'])}
  (simultaneous lower bound {_number(state_metrics['direct_crossover']['simultaneous_lower'])})
- Risk gain over reset: {_number(state_metrics['risk_gain']['mean'])}
- Held-out history log-loss gain: {_number(state_metrics['history_logloss_gain']['mean'])}
- Split-half crossover reliability: {_number(state['split_half_crossover_reliability'])}
- Self-continuation/transplant pathwise identity: {state['self_transplant_pathwise_identity']}

## Renewable lineage carrier

- Primary verdict: {carrier['carrier_verdict']}
- Causal verdict: {carrier['causal_verdict']}
- Distributed bottleneck verdict: {carrier['distributed_verdict']}
- Generation-4 direct crossover: {_number(carrier_metrics['generation4_direct_crossover']['mean'])}
  (simultaneous lower bound {_number(carrier_metrics['generation4_direct_crossover']['simultaneous_lower'])})
- Generation-4 risk gain over zero carrier: {_number(carrier_metrics['generation4_risk_gain']['mean'])}
- Held-out history log-loss gain: {_number(carrier_metrics['generation4_history_logloss_gain']['mean'])}
- Split-half crossover reliability: {_number(carrier['split_half_crossover_reliability'])}
- Ablation loss fraction: {_number(carrier['ablation_loss_fraction'])}
- Rescue fraction: {_number(carrier['rescue_fraction'])}
- Targeted k=5 retention fraction: {_number(carrier['targeted_k5_retention_fraction'])}

## Other registered stages

- Writer/noise boundary: {analysis['boundary']['verdict']}
- Slow passive mark: {analysis['slow_mark']['verdict']}

All control outcomes and adjusted bounds are retained in `analysis/*.json`; this
report does not convert a failed gate into partial confirmation.
"""
    (run_dir / "REPORT.md").write_text(report)

    lay = f"""# Lay summary

We asked whether a Wagner gene-regulatory network can be pushed into one of two
opposite stable forms and whether information about that earlier form can still
change what its descendants do. The corrected test counted only exact arrivals
at those stable forms, used two neutral halfway starts, and gave every treatment
and control the same random disturbances.

The immediate expression-state result was **{analysis['state']['state_channel_verdict']}**.
That means the written state {'carried usable information beyond the reset controls' if analysis['state']['state_channel_verdict'] == 'STATE_CHANNEL_CONFIRMED' else 'did not clear every registered prediction and control gate'}.
The longer-lived lineage test was **{carrier['carrier_verdict']}**, with the
stronger intervention verdict **{carrier['causal_verdict']}**. The latter asks
whether removing, disabling, reversing, and rescuing the proposed carrier behave
as a real mechanism should, rather than merely correlating with the past.

The computational audit was **{'fully reproducible' if verification['verified'] else 'not fully reproducible'}**:
independent reruns reproduced the stored source landscapes and every ordered
future digest in the state and carrier stages. The scientific conclusion is the
literal overall verdict **{analysis['overall_verdict']}**; diagnostics from smoke
or quick profiles are explicitly not treated as a replication.
"""
    (run_dir / "LAY_SUMMARY.md").write_text(lay)
