# P3 / CR4 beta-surgery pilot

## Outcome

Pilot eligibility for a later untouched confirmation: **False**.
The original full four-cell confirmatory gate passed: **False**.
Complete deterministic replay passed: **True**.

A pilot eligibility result is developmental evidence only. It is not the separately registered 160-matrix confirmation and cannot establish cross-clean-room replication.

## Primary cells

| Cell | Up−down | 95% matrix-bootstrap CI | Holm p | Random−no-op | Full gate |
|---|---:|---:|---:|---:|---:|
| c02_A | -0.003125 | [-0.021445, 0.016250] | 0.848914 | 0.006875 | False |
| c02_B | 0.003125 | [-0.020313, 0.026875] | 0.848914 | 0.008750 | False |
| c03_A | 0.018125 | [0.000625, 0.035000] | 0.0956798 | -0.012500 | False |
| c03_B | 0.010312 | [-0.013438, 0.033438] | 0.630461 | -0.006875 | False |

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
