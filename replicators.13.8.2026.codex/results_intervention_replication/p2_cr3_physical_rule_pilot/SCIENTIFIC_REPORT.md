# P2 / CR3 catalytic-support rule pilot

## Outcome

Pilot eligibility for a later untouched confirmation: **False**.
The original full four-cell confirmatory gate passed: **False**.
Complete deterministic replay passed: **True**.

A pilot eligibility result is developmental evidence only. It is not the separately registered 160-matrix confirmation and cannot establish cross-clean-room replication.

## Primary cells

| Cell | Up−down | 95% matrix-bootstrap CI | Holm p | Random−no-op | Full gate |
|---|---:|---:|---:|---:|---:|
| c02_A | -0.004688 | [-0.021133, 0.011562] | 1 | 0.007187 | False |
| c02_B | 0.026875 | [0.001250, 0.051875] | 0.0732243 | 0.014062 | False |
| c03_A | -0.000312 | [-0.016250, 0.016250] | 1 | -0.008750 | False |
| c03_B | 0.029375 | [0.009687, 0.050313] | 0.0195265 | -0.008750 | False |

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
