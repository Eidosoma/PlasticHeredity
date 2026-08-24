# Prospective inheritance-dependence confirmation

## Outcome

Both first-order and registered duration-aware dependence passed the prospective gates in both simulator candidates.

| Contrast | Candidate | Pooled gain (bits/transition) | 95% CI | Holm p | Both directions positive | Pass |
|---|---:|---:|---:|---:|---:|---:|
| markov_vs_iid | 02 | 0.046953 | [0.039503, 0.056486] | 0.000976 | True | True |
| semimarkov_vs_markov | 02 | 0.010770 | [0.008657, 0.013493] | 0.000976 | True | True |
| markov_vs_iid | 03 | 0.033936 | [0.028894, 0.040376] | 0.000976 | True | True |
| semimarkov_vs_markov | 03 | 0.009984 | [0.007921, 0.012333] | 0.000976 | True | True |

## Sequence support

| Candidate | Futures | No break | Empty suffix | Singleton suffix | Usable suffixes | Scored transitions |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 64000 | 16523 | 352 | 382 | 46743 | 1030942 |
| 03 | 64000 | 14661 | 292 | 327 | 48720 | 1095600 |

## Claim boundary

A passing result means that recent binary inheritance state, and if applicable registered run duration, improves out-of-matrix next-symbol prediction over the nested baseline. It does not by itself establish biological memory, molecular information storage, error correction, or a causal mechanism; latent state and matrix heterogeneity may contribute to the predictive dependence.

This memory analysis is separate from, and does not alter, the confirmed 12-fission L54 break-and-renewal risk predictor.

## Audit boundary

Registration `0a100eb3d626f3fdb92f5b4f84f1404b095fc1d21b1dfe6b3a83d2adf0e78f1f` was sealed before `MEMCONF`. All 128000 futures were regenerated exactly: **True**.
