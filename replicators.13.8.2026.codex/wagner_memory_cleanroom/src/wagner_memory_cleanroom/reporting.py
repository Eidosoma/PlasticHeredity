from __future__ import annotations

from pathlib import Path
from typing import Any

from .storage import atomic_json


def _number(value: Any, digits: int = 4) -> str:
    return "not estimable" if value is None else f"{float(value):.{digits}f}"


def write_reports(run_dir: Path, analysis: dict[str, Any], verification: dict[str, Any]) -> None:
    state = analysis["state"]
    carrier = analysis["carrier"]
    verdict = {
        "format": "wagner-memory-final-verdict-v1",
        "scientific": bool(analysis["scientific"]),
        "overall_verdict": analysis["overall_verdict"],
        "state_channel_verdict": state["state_channel_verdict"],
        "soft_writer_verdict": state["soft_writer_verdict"],
        "noise_boundary_verdict": analysis["boundary"]["verdict"],
        "slow_mark_verdict": analysis["slow_mark"]["verdict"],
        "carrier_verdict": carrier["carrier_verdict"],
        "causal_verdict": carrier["causal_verdict"],
        "distributed_verdict": carrier["distributed_verdict"],
        "independent_regeneration_verified": bool(verification["independent_regeneration_verified"]),
        "replay_verified": bool(verification["replay_verified"]),
    }
    atomic_json(run_dir / "FINAL_VERDICT.json", verdict)

    hard = state["writers"]["hard-theta-0"]
    soft = state["writers"]["soft-theta-0"]
    report = f"""# Clean-room Wagner memory replication

Overall verdict: **{analysis['overall_verdict']}**

| Tier | Verdict | Primary risk gain | Lower bound | Reliability |
|---|---|---:|---:|---:|
| Hard expression-state transfer | {hard['verdict']} | {_number(hard['risk_gain']['mean'])} | {_number(hard['risk_gain']['lower'])} | {_number(hard['split_half_reliability'])} |
| Soft expression writer | {soft['verdict']} | {_number(soft['risk_gain']['mean'])} | {_number(soft['risk_gain']['lower'])} | {_number(soft['split_half_reliability'])} |
| Full renewable latch | {carrier['carrier_verdict']} | {_number(carrier['generation4_risk_gain']['mean'])} | {_number(carrier['generation4_risk_gain']['lower'])} | {_number(carrier['split_half_reliability'])} |

## Separate verdicts

- Noise boundary: **{analysis['boundary']['verdict']}**
- Decaying slow mark: **{analysis['slow_mark']['verdict']}**
- Carrier causality: **{carrier['causal_verdict']}**
- Distributed carrier: **{carrier['distributed_verdict']}**
- Generation-2 ablation loss fraction: {_number(carrier['ablation_loss_fraction'])}
- Generation-3 rescue fraction: {_number(carrier['rescue_fraction'])}
- Targeted k=5 retention fraction: {_number(carrier['targeted_k5_retention_fraction'])}

## Integrity

- Independent full state/carrier regeneration: {verification['independent_regeneration_verified']}
- Frozen replay audit: {verification['replay_verified']}
- Scientific profile: {analysis['scientific']}

## Interpretation boundary

A positive carrier result establishes an engineered in-silico Wagner carrier
that is written from founder expression and renewed across complete expression
resets. It does not identify a naturally evolved or biological epigenetic
mechanism. F12 prediction, Boolean networks, evolution, CA, and manuscript work
were outside this campaign.
"""
    (run_dir / "REPORT.md").write_text(report)

    lay = f"""# Lay summary

This clean-room test asked whether a small gene-regulatory network can remember
which of two temporary histories it experienced. The already completed F12
prediction experiment was not rerun. Here the question was transmission: does
the present gene state carry history, and can a separate memory survive when
every new generation starts with its gene-expression state erased?

The direct state-transfer verdict was **{state['state_channel_verdict']}**. In
that test, the acquired expression pattern was moved into a genetically
identical recipient and compared with a neutral reset, a shuffled pattern, and
matched destination controls. The soft writer and noise series were kept as
separate boundary tests rather than being allowed to replace the primary hard
state result.

The multigeneration latch verdict was **{carrier['carrier_verdict']}**, with the
causal-control verdict **{carrier['causal_verdict']}**. Children inherited no
adult gene expression: they received only a ten-entry latch, developed from the
same neutral state, rewrote the latch, and handed that latch onward. Shuffling,
zeroing, disabling reading or writing, ablation, rescue, opposite history, and
carrier bottlenecks tested whether the result genuinely depended on that
renewed memory channel.

The combined verdict was **{analysis['overall_verdict']}**. Even a pass is an
engineered simulation result, not evidence that real genes use this particular
latch. Independent regeneration and replay were {'successful' if verification['independent_regeneration_verified'] else 'not successful'}, and all failed gates remain visible in the machine-readable analyses.
"""
    (run_dir / "LAY_SUMMARY.md").write_text(lay)

