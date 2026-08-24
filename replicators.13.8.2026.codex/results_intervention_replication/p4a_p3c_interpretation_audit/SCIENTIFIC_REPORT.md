# P3c sealed-data interpretation audit

This audit generated no matrices or futures and does not alter P3c's registered failure.

## Gate attribution

| Cell | Loosen-tighten | Neutral-noop | Neutral 90% CI | Failed gates |
|---|---:|---:|---:|---|
| c02_A | +0.113359 | +0.010313 | [-0.001094, +0.021328] | none |
| c02_B | +0.103359 | +0.015000 | [+0.003496, +0.026406] | neutral_tost_equivalent |
| c03_A | +0.108984 | +0.015312 | [+0.003281, +0.027266] | neutral_tost_equivalent |
| c03_B | +0.104453 | +0.023438 | [+0.012500, +0.034219] | neutral_tost_equivalent |

All four targeted strength contrasts, confidence bounds, randomization tests, throughput slopes, and rank associations passed. The confirmation failed because the fixed-throughput topology arm was not equivalent to NOOP in three cells.

## Process interpretation

- c02_A: fixed-throughput topology changed JOINT_BREAK_RUN3 by +0.0103, first-break probability by +0.0120, and mean growth updates per fission by -7.197.
- c02_B: fixed-throughput topology changed JOINT_BREAK_RUN3 by +0.0150, first-break probability by +0.0180, and mean growth updates per fission by -7.256.
- c03_A: fixed-throughput topology changed JOINT_BREAK_RUN3 by +0.0153, first-break probability by +0.0154, and mean growth updates per fission by -5.450.
- c03_B: fixed-throughput topology changed JOINT_BREAK_RUN3 by +0.0234, first-break probability by +0.0264, and mean growth updates per fission by -5.307.

The neutral arm preserved only the launch-state scalar x^T beta x. It changed how catalytic support was distributed across types, which changed subsequent kinetics and composition. Thus total starting throughput is a strong causal axis but not a sufficient description of the network.

## Boundary

Geometry/outcome associations are exploratory and do not establish mediation. No analysis conditions on an intervention-created break. Causal renewal awaits a fresh identical-natural-break experiment.
