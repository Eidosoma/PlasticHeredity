# P1 / CR1 predictor-guided molecular intervention pilot

## Outcome

Pilot eligibility for a later untouched confirmation: **True**.
The original full four-cell confirmatory gate passed: **False**.
Complete deterministic replay passed: **True**.

A pilot eligibility result is developmental evidence only. It is not the separately registered 160-matrix confirmation and cannot establish cross-clean-room replication.

## Primary cells

| Cell | Up−down | 95% matrix-bootstrap CI | Holm p | Random−no-op | Full gate |
|---|---:|---:|---:|---:|---:|
| c02_A | 0.097812 | [0.066055, 0.129375] | 0.000976324 | -0.001250 | True |
| c02_B | 0.098750 | [0.066250, 0.132188] | 0.000976324 | 0.021875 | False |
| c03_A | 0.103125 | [0.069687, 0.135938] | 0.000976324 | 0.005938 | False |
| c03_B | 0.093125 | [0.063750, 0.122695] | 0.000976324 | -0.005312 | True |

The inference unit was the catalytic matrix. All landmarks, arms, fixed branch halves, and repeated states from a matrix remained together. Bootstrap and sign-randomization draws were shared across all four cells.

## Design and audit

- Matrices: 40 fresh matrices shared across candidates.
- Restored states: 400.
- Futures per pass: 51,200 F12 futures.
- Every scientific future was replayed; futures were never retried.
- Paired arms used common random streams whose seed key omitted arm identity.
- Registration ID: `f61e0340dcd8c9ae6b606c8133ca3d8fb1de2e13fe863719aa67b649e8b74531`.

## Claim boundary

This pilot tests causal movement of the operational break-and-renewal probability under the registered intervention family. It does not test Phi/PhiID, strict-eight control, biological memory, autonomy, life, real chemistry, or a universal origin-of-life mechanism. A positive pilot does not by itself establish a common confirmed control law.

## Mandatory stop

This stage is sealed and the workflow stops here. No subsequent scientific pilot or confirmation is launched without a new user instruction.
