# CR9 control half-life and minimum feedback rate

Registered two-candidate accumulating-hysteresis gate: **False**.
Complete gate including replay/no-op/readback integrity: **False**.

## Steering-pulse ladder

| Candidate | Mean matrix Spearman | 95% matrix-bootstrap CI | One-sided randomization p | Primary gate |
|---|---:|---:|---:|---:|
| 02 | +0.140725 | (-0.01591756419463154, 0.2910563970180997) | 0.0375885 | False |
| 03 | +0.108220 | (-0.03909876981785691, 0.24942709319632964) | 0.0744447 | False |

Mean persistence (fissions before anchor similarity first falls below 0.7; cap 61):

| Candidate | Pulse 1 | Pulse 2 | Pulse 4 | Pulse 8 | Pulse 16 | Pulse 32 | Pulse 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 02 | 12.566 | 13.708 | 13.847 | 11.802 | 12.712 | 11.698 | 11.302 |
| 03 | 13.333 | 14.122 | 11.962 | 11.490 | 11.542 | 10.969 | 11.323 |

## Periodic active feedback

The descriptive minimum-feedback interval is the largest registered K whose MODEL_EVERY_K 95% lower bound is positive against both budget-matched random editing and NOOP. It is not a confirmatory rescue gate.

| Candidate | Descriptive largest supported interval |
|---|---:|
| 02 | 16 |
| 03 | 16 |

## Event-triggered active feedback

| Candidate | Policy | Inheritance | Mean edits/60 | Gain vs NOOP | 95% CI | Fraction continuous gain |
|---|---|---:|---:|---:|---:|---:|
| 02 | THRESHOLD_015 | 0.992245 | 23.931 | +0.075231 | (0.05386284722222223, 0.09992042824074074) | 0.969 |
| 02 | THRESHOLD_025 | 0.988079 | 17.462 | +0.071065 | (0.050021701388888895, 0.09581163194444445) | 0.916 |
| 02 | THRESHOLD_035 | 0.982870 | 12.455 | +0.065856 | (0.0453125, 0.08998842592592589) | 0.849 |
| 02 | CONTINUOUS | 0.994618 | 60.000 | +0.077604 | (0.056387442129629636, 0.10235098379629631) | 1.000 |
| 03 | THRESHOLD_015 | 0.990914 | 28.406 | +0.085185 | (0.06226851851851853, 0.11141493055555556) | 0.981 |
| 03 | THRESHOLD_025 | 0.988889 | 21.410 | +0.083160 | (0.060474537037037035, 0.10894820601851853) | 0.957 |
| 03 | THRESHOLD_035 | 0.984664 | 16.285 | +0.078935 | (0.05613425925925928, 0.10524450231481483) | 0.909 |
| 03 | CONTINUOUS | 0.992593 | 60.000 | +0.086863 | (0.06406250000000001, 0.11276765046296297) | 1.000 |

All inference treats the catalytic matrix as the unit and keeps candidates separate. Missing registered boundaries after extinction count adversely in the fixed-horizon inheritance outcome. Full schedules, contrasts, action records, state trajectories, bootstrap draws, sign randomizations, and replay audits are machine-readable alongside this report.

A longer-lived post-control trace would be transient hysteresis, not an autonomous restoring basin. Periodic and triggered policies remain active external feedback, even when they use few edits.
