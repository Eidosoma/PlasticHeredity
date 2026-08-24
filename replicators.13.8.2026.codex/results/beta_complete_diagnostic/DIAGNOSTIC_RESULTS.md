# Post-hoc beta-completeness diagnostic

## Outcome

The registered current-composition contrast did not pass all gates.
No incremental static-beta signal was detected with the frozen comprehensive threshold-free panel.
The beta-conditioned current-state block did not pass beyond the comprehensive additive baseline.

## Primary registered contrasts

| Contrast | Candidate | Half | Log-loss gain | 95% CI | Holm p | Pass |
|---|---:|---:|---:|---:|---:|---:|
| state | 02 | A | 0.001848 | [0.000198, 0.003605] | 0.182573 | False |
| state | 02 | B | 0.001776 | [0.000098, 0.003381] | 0.224555 | False |
| network | 02 | A | 0.000005 | [-0.001218, 0.001457] | 1.000000 | False |
| network | 02 | B | -0.000434 | [-0.001664, 0.000839] | 1.000000 | False |
| interaction | 02 | A | -0.029305 | [-0.059262, -0.004826] | 1.000000 | False |
| interaction | 02 | B | -0.029009 | [-0.061698, -0.003948] | 1.000000 | False |
| state | 03 | A | 0.001388 | [-0.001850, 0.004525] | 1.000000 | False |
| state | 03 | B | 0.002112 | [-0.000838, 0.005184] | 0.701001 | False |
| network | 03 | A | 0.002242 | [0.000319, 0.004487] | 0.224555 | False |
| network | 03 | B | 0.002559 | [0.000473, 0.004829] | 0.090798 | False |
| interaction | 03 | A | -0.007637 | [-0.017899, 0.000089] | 1.000000 | False |
| interaction | 03 | B | -0.005564 | [-0.013742, 0.000383] | 1.000000 | False |

## Frozen representation

| Candidate | Retained state | Retained beta | Retained interaction | Lambda state/beta/interaction |
|---|---:|---:|---:|---|
| 02 | 11 | 240 | 64 | 0.1 / 10 / 10 |
| 03 | 12 | 240 | 64 | 0.01 / 1 / 10 |

No added block uses PCA. The beta panel is threshold-free and includes the complete normalized singular spectrum. A null beta result is representation-specific and is not proof that beta is generally uninformative.

## Audit boundary

This applies a previously sealed correction to an already-seen cohort and cannot support a new prospective claim.

Registration: `7c6acd1e3bac96dae7931c48bff39deedfc6550344dfea3753949956f71701bd`. Exact future replay: **not applicable**. Independent gain recomputation within 1e-14: **True**.

These are predictive, simulator-specific contrasts. They do not establish a causal mechanism or biological chemistry.
