# P3b beta-surgery dose bridge — singleton-recovered result

## Recovery disclosure

The original run stopped after 363 states because a balanced random surgery is undefined on a one-entry present-present block. The recovery was sealed before replay or inference. All 4 structurally ineligible states remained in the cohort and used unchanged beta in every arm, producing a zero paired contribution.

Original registration: `c1fe38be6a7e2b71eb5e288c9e238ff45c30d8fe388f2d21a879acea6dd5624e`. Recovery amendment: `c5a941f9a6b1d71fcd02b4d3878cb9c091cc2090bd914273d58f5740d3e183c0`.

## Registered outcome

Landmark-60 Fable-strength replication gate: **False**.
Five-landmark generalization gate: **False**.
Landmark-60 two-dose ordering gate: **True**.
Five-landmark two-dose ordering gate: **True**.
Exact replay: **True**.

## Primary landmark 60

| Cell | Loosen−tighten | 95% CI | Holm p | Random 90% CI | Pass |
|---|---:|---:|---:|---:|---:|
| c02_A | 0.096875 | [0.064355, 0.128906] | 0.000976324 | [-0.067969, -0.000781] | False |
| c02_B | 0.121094 | [0.080762, 0.162500] | 0.000976324 | [-0.017188, 0.045312] | False |
| c03_A | 0.089063 | [0.056250, 0.122656] | 0.000976324 | [-0.029687, 0.035156] | False |
| c03_B | 0.124219 | [0.089844, 0.157812] | 0.000976324 | [-0.077344, -0.013281] | False |

## Five-landmark generalization

| Cell | Loosen−tighten | 95% CI | Holm p | Random 90% CI | Pass |
|---|---:|---:|---:|---:|---:|
| c02_A | 0.122188 | [0.099590, 0.146094] | 0.000976324 | [-0.025781, 0.000977] | False |
| c02_B | 0.106094 | [0.081406, 0.131406] | 0.000976324 | [-0.032188, -0.001719] | False |
| c03_A | 0.103438 | [0.081094, 0.126562] | 0.000976324 | [-0.027813, -0.001875] | False |
| c03_B | 0.095000 | [0.071719, 0.118125] | 0.000976324 | [-0.028164, 0.001719] | False |

## Audit

- Structural no-action audit: **True**.
- Eligible surgery states: 956.
- Structural no-action states: 4.
- Every matrix and state was retained; none was replaced or retried.
- Reused original generation checkpoints: 363, unchanged byte-for-byte.
- Every state and arm was included in the complete deterministic replay.

## Boundary and stop

The estimand is the registered natural-cohort policy with structural no-action where a matched balanced random control is undefined. This cannot establish life, biological memory, autonomy, real chemistry, Phi/PhiID intervention, or strict-eight control. The result is sealed and no later phase launches automatically.
