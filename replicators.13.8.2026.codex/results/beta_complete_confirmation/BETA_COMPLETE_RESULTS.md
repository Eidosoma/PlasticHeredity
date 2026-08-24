# Prospective beta-completeness confirmation

## Outcome

The registered current-composition contrast did not pass all gates.
No incremental static-beta signal was detected with the frozen comprehensive threshold-free panel.
The beta-conditioned current-state block did not pass beyond the comprehensive additive baseline.

## Primary registered contrasts

| Contrast | Candidate | Half | Log-loss gain | 95% CI | Holm p | Pass |
|---|---:|---:|---:|---:|---:|---:|
| state | 02 | A | 0.001419 | [0.000005, 0.002902] | 0.270930 | False |
| state | 02 | B | 0.001282 | [-0.000169, 0.002754] | 0.412985 | False |
| network | 02 | A | -0.000226 | [-0.001153, 0.000772] | 1.000000 | False |
| network | 02 | B | 0.000037 | [-0.000847, 0.000935] | 1.000000 | False |
| interaction | 02 | A | -0.009269 | [-0.020761, -0.000939] | 1.000000 | False |
| interaction | 02 | B | -0.009987 | [-0.022302, -0.001552] | 1.000000 | False |
| state | 03 | A | 0.002216 | [-0.000660, 0.005085] | 0.523310 | False |
| state | 03 | B | 0.001577 | [-0.001381, 0.004455] | 1.000000 | False |
| network | 03 | A | 0.001546 | [0.000105, 0.003064] | 0.210886 | False |
| network | 03 | B | 0.001475 | [0.000051, 0.003020] | 0.214791 | False |
| interaction | 03 | A | -0.001738 | [-0.004476, 0.000931] | 1.000000 | False |
| interaction | 03 | B | -0.001221 | [-0.004012, 0.001349] | 1.000000 | False |

## Frozen representation

| Candidate | Retained state | Retained beta | Retained interaction | Lambda state/beta/interaction |
|---|---:|---:|---:|---|
| 02 | 11 | 240 | 64 | 0.1 / 10 / 10 |
| 03 | 12 | 240 | 64 | 0.01 / 1 / 10 |

No added block uses PCA. The beta panel is threshold-free and includes the complete normalized singular spectrum. A null beta result is representation-specific and is not proof that beta is generally uninformative.

## Audit boundary

The registration was sealed before any MECHCONF2 matrix was generated.

Registration: `7c6acd1e3bac96dae7931c48bff39deedfc6550344dfea3753949956f71701bd`. Exact future replay: **True**. Independent gain recomputation within 1e-14: **True**.

These are predictive, simulator-specific contrasts. They do not establish a causal mechanism or biological chemistry.
