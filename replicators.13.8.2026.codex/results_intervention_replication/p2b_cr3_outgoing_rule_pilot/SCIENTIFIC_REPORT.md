# P2b / corrected Fable C3 outgoing-rule pilot

## Outcome

Pilot eligibility: **False**.
Full four-cell gate: **False**.
Exact replay: **True**.

P2b uses `x @ beta` (`beta.T @ x`), the externally disambiguated frozen Fable C3 source/outgoing influence. The original sealed P2 used `beta @ x` and remains an incoming-support negative control.

| Cell | Up−down | 95% matrix-bootstrap CI | Holm p | Random−no-op | Full gate |
|---|---:|---:|---:|---:|---:|
| c02_A | 0.097187 | [0.068867, 0.127383] | 0.000976324 | -0.013125 | False |
| c02_B | 0.091250 | [0.065312, 0.117500] | 0.000976324 | -0.028750 | False |
| c03_A | 0.090938 | [0.061250, 0.121875] | 0.000976324 | 0.010625 | False |
| c03_B | 0.083125 | [0.052187, 0.115937] | 0.000976324 | -0.001562 | True |

The catalytic matrix was the inference unit. Branch halves and candidates were evaluated separately with shared whole-matrix draws.

## Design and audit

- 40 fresh matrices, two candidates, five landmarks, and 400 states.
- 51,200 primary F12 futures and complete deterministic replay.
- Common random streams across arms; arm identity absent from future seeds.
- Registration: `02185a2364e03628c0417f937548c518c0c8c0025514809a7bb3edef25be0a23`.

## Boundary

This is a developmental correction pilot, not an untouched confirmation. It cannot establish life, biological memory, autonomy, Phi/PhiID, real chemistry, or a universal origin-of-life mechanism.

## Mandatory stop

The result is sealed. P3 and confirmation remain unlaunched pending review.
