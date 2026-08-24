# GN1 catalytic-versus-geometric null decomposition

GN1 is a prospectively registered reviewer-response decomposition with no single omnibus pass gate. All rates and effects are candidate-separated and use whole catalytic matrices for inference.

## Event prevalence and statewise reliability

| Candidate | Mechanism | F12 | 95% CI | strict all8 F32 | 95% CI | reliability | centered reliability |
|---|---|---:|---:|---:|---:|---:|---:|
| 02 | NATURAL_GARD | 0.3758 | [+0.3343, +0.4189] | 0.0186 | [+0.0118, +0.0264] | +0.922 | +0.688 |
| 02 | HOMOGENEOUS_GENERATIVE | 0.0000 | [+0.0000, +0.0000] | 0.0000 | [+0.0000, +0.0000] | +nan | +nan |
| 02 | COUPLING_DERANGED | 0.4870 | [+0.4499, +0.5255] | 0.0325 | [+0.0201, +0.0473] | +0.900 | +0.758 |
| 02 | FISSION_ONLY_GENERATIVE | 0.0000 | [+0.0000, +0.0000] | 0.0000 | [+0.0000, +0.0000] | +nan | +nan |
| 03 | NATURAL_GARD | 0.4144 | [+0.3757, +0.4513] | 0.0184 | [+0.0124, +0.0256] | +0.883 | +0.702 |
| 03 | HOMOGENEOUS_GENERATIVE | 0.0000 | [+0.0000, +0.0000] | 0.0000 | [+0.0000, +0.0000] | +nan | +nan |
| 03 | COUPLING_DERANGED | 0.5329 | [+0.4990, +0.5666] | 0.0380 | [+0.0239, +0.0547] | +0.863 | +0.670 |
| 03 | FISSION_ONLY_GENERATIVE | 0.0000 | [+0.0000, +0.0000] | 0.0000 | [+0.0000, +0.0000] | +nan | +nan |

Overall reliability includes stable between-matrix propensities. The centered value removes each matrix mean and is the more direct test of landmark-specific state dependence.

## Natural-minus-null event differences

| Endpoint | Candidate | Null | Natural-null | 95% CI | Holm p |
|---|---|---|---:|---:|---:|
| f12 | 02 | HOMOGENEOUS_GENERATIVE | +0.3758 | [+0.3351, +0.4177] | 0.00146449 |
| f12 | 02 | COUPLING_DERANGED | -0.1112 | [-0.1451, -0.0769] | 1 |
| f12 | 02 | FISSION_ONLY_GENERATIVE | +0.3758 | [+0.3356, +0.4163] | 0.00146449 |
| f12 | 03 | HOMOGENEOUS_GENERATIVE | +0.4144 | [+0.3745, +0.4529] | 0.00146449 |
| f12 | 03 | COUPLING_DERANGED | -0.1185 | [-0.1512, -0.0864] | 1 |
| f12 | 03 | FISSION_ONLY_GENERATIVE | +0.4144 | [+0.3747, +0.4524] | 0.00146449 |
| strict_all8 | 02 | HOMOGENEOUS_GENERATIVE | +0.0186 | [+0.0118, +0.0264] | 0.00146449 |
| strict_all8 | 02 | COUPLING_DERANGED | -0.0140 | [-0.0238, -0.0053] | 1 |
| strict_all8 | 02 | FISSION_ONLY_GENERATIVE | +0.0186 | [+0.0120, +0.0265] | 0.00146449 |
| strict_all8 | 03 | HOMOGENEOUS_GENERATIVE | +0.0184 | [+0.0124, +0.0255] | 0.00146449 |
| strict_all8 | 03 | COUPLING_DERANGED | -0.0196 | [-0.0337, -0.0076] | 1 |
| strict_all8 | 03 | FISSION_ONLY_GENERATIVE | +0.0184 | [+0.0122, +0.0255] | 0.00146449 |

## Frozen manuscript-algorithm transfer

Positive values mean lower branch-level log loss for FULL_STATE_GRAPH_HISTORY than DIRECT_HISTORY_PHASE.

| Candidate | Mechanism | Half | Log-loss gain | 95% CI | Holm p |
|---|---|---|---:|---:|---:|
| 02 | NATURAL_GARD | A | +0.025581 | [+0.0180, +0.0336] | 0.00195265 |
| 02 | HOMOGENEOUS_GENERATIVE | A | -13.850090 | [-13.8867, -13.8144] | 1 |
| 02 | COUPLING_DERANGED | A | +0.069123 | [+0.0520, +0.0864] | 0.00195265 |
| 02 | FISSION_ONLY_GENERATIVE | A | -13.901461 | [-13.9393, -13.8651] | 1 |
| 03 | NATURAL_GARD | A | +0.025480 | [+0.0165, +0.0347] | 0.00195265 |
| 03 | HOMOGENEOUS_GENERATIVE | A | +3.967953 | [+3.9064, +4.0315] | 0.00195265 |
| 03 | COUPLING_DERANGED | A | +0.076788 | [+0.0599, +0.0939] | 0.00195265 |
| 03 | FISSION_ONLY_GENERATIVE | A | +3.949531 | [+3.8884, +4.0116] | 0.00195265 |
| 02 | NATURAL_GARD | B | +0.024272 | [+0.0165, +0.0323] | 0.00195265 |
| 02 | HOMOGENEOUS_GENERATIVE | B | -13.850090 | [-13.8850, -13.8147] | 1 |
| 02 | COUPLING_DERANGED | B | +0.069835 | [+0.0536, +0.0869] | 0.00195265 |
| 02 | FISSION_ONLY_GENERATIVE | B | -13.901461 | [-13.9382, -13.8654] | 1 |
| 03 | NATURAL_GARD | B | +0.024821 | [+0.0161, +0.0337] | 0.00195265 |
| 03 | HOMOGENEOUS_GENERATIVE | B | +3.967953 | [+3.9062, +4.0319] | 0.00195265 |
| 03 | COUPLING_DERANGED | B | +0.070349 | [+0.0529, +0.0883] | 0.00195265 |
| 03 | FISSION_ONLY_GENERATIVE | B | +3.949531 | [+3.8864, +4.0121] | 0.00195265 |

## Null-specific clean composition predictor

Positive values mean that a two-way whole-matrix-cross-fitted composition block improves over the unique H10 direct baseline. Penalties were fixed before outcomes and no null-specific tuning occurred.

| Candidate | Mechanism | Half | Log-loss gain | 95% CI | Holm p |
|---|---|---|---:|---:|---:|
| 02 | NATURAL_GARD | A | +0.002500 | [-0.0004, +0.0055] | 0.391262 |
| 02 | HOMOGENEOUS_GENERATIVE | A | +0.000000 | [+0.0000, +0.0000] | 1 |
| 02 | COUPLING_DERANGED | A | +0.002507 | [-0.0011, +0.0060] | 0.497925 |
| 02 | FISSION_ONLY_GENERATIVE | A | +0.000000 | [+0.0000, +0.0000] | 1 |
| 03 | NATURAL_GARD | A | +0.003953 | [+0.0006, +0.0078] | 0.148401 |
| 03 | HOMOGENEOUS_GENERATIVE | A | +0.000000 | [+0.0000, +0.0000] | 1 |
| 03 | COUPLING_DERANGED | A | +0.002229 | [-0.0017, +0.0060] | 0.684647 |
| 03 | FISSION_ONLY_GENERATIVE | A | +0.000000 | [+0.0000, +0.0000] | 1 |
| 02 | NATURAL_GARD | B | +0.003502 | [+0.0006, +0.0066] | 0.0781059 |
| 02 | HOMOGENEOUS_GENERATIVE | B | +0.000000 | [+0.0000, +0.0000] | 1 |
| 02 | COUPLING_DERANGED | B | +0.002287 | [-0.0008, +0.0053] | 0.44081 |
| 02 | FISSION_ONLY_GENERATIVE | B | +0.000000 | [+0.0000, +0.0000] | 1 |
| 03 | NATURAL_GARD | B | +0.003788 | [+0.0002, +0.0077] | 0.193068 |
| 03 | HOMOGENEOUS_GENERATIVE | B | +0.000000 | [+0.0000, +0.0000] | 1 |
| 03 | COUPLING_DERANGED | B | -0.000911 | [-0.0047, +0.0029] | 1 |
| 03 | FISSION_ONLY_GENERATIVE | B | +0.000000 | [+0.0000, +0.0000] | 1 |

## Transported outgoing-rule intervention

The original heterogeneous beta selected SOURCE_RULE_UP and SOURCE_RULE_DOWN in every mechanism. Positive effects mean more F12 under the risk-raising than the stabilizing edit.

| Candidate | Mechanism | Up-down | 95% CI | Holm p | Null TOST | Random-noop | Random TOST |
|---|---|---:|---:|---:|---:|---:|---:|
| 02 | NATURAL_GARD | +0.0949 | [+0.0799, +0.1109] | 0.00195265 | n/a | +0.0062 | True |
| 02 | HOMOGENEOUS_GENERATIVE | +0.0000 | [+0.0000, +0.0000] | 1 | True | +0.0000 | True |
| 02 | COUPLING_DERANGED | +0.0018 | [-0.0091, +0.0122] | 1 | True | -0.0018 | True |
| 02 | FISSION_ONLY_GENERATIVE | +0.0000 | [+0.0000, +0.0000] | 1 | True | +0.0000 | True |
| 03 | NATURAL_GARD | +0.0944 | [+0.0781, +0.1121] | 0.00195265 | n/a | +0.0016 | True |
| 03 | HOMOGENEOUS_GENERATIVE | +0.0000 | [+0.0000, +0.0000] | 1 | True | +0.0000 | True |
| 03 | COUPLING_DERANGED | -0.0094 | [-0.0214, +0.0029] | 1 | True | -0.0067 | True |
| 03 | FISSION_ONLY_GENERATIVE | +0.0000 | [+0.0000, +0.0000] | 1 | True | +0.0000 | True |

## Integrity and claim boundary

Every scientific future was replayed completely. Natural no-op uses the unmodified sealed simulator. No future was retried and no failed or extinct matrix was replaced.

GN1 can quantify a geometric floor and catalytic contributions within reconstructed GARD. It cannot establish life, biological memory, autonomous agency, an installed compotype, real prebiotic chemistry, Phi/PhiID, or a universal origin-of-life mechanism.
