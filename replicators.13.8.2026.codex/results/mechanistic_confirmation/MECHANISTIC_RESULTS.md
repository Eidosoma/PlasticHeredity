# Prospective mechanistic-ablation confirmation

## Outcome

The mass-free network-conditioned state interaction added prospective information beyond the additive history + state + network model.

Supported registered contrasts: **state, interaction**.

## Primary gates

| Contrast | Candidate | Half | Log-loss gain | 95% CI | Holm p | Pass |
|---|---:|---:|---:|---:|---:|---:|
| state | 02 | A | 0.00646 | [0.00368, 0.00930] | 0.00293 | True |
| state | 02 | B | 0.00676 | [0.00395, 0.00954] | 0.00293 | True |
| network | 02 | A | -0.00008 | [-0.00229, 0.00225] | 0.81328 | False |
| network | 02 | B | 0.00081 | [-0.00140, 0.00312] | 0.81328 | False |
| interaction | 02 | A | 0.00968 | [0.00587, 0.01358] | 0.00293 | True |
| interaction | 02 | B | 0.01021 | [0.00674, 0.01372] | 0.00293 | True |
| state | 03 | A | 0.00853 | [0.00357, 0.01298] | 0.00293 | True |
| state | 03 | B | 0.00726 | [0.00239, 0.01167] | 0.00293 | True |
| network | 03 | A | 0.00073 | [-0.00150, 0.00303] | 0.81328 | False |
| network | 03 | B | 0.00097 | [-0.00135, 0.00337] | 0.81328 | False |
| interaction | 03 | A | 0.00854 | [0.00380, 0.01394] | 0.00293 | True |
| interaction | 03 | B | 0.01031 | [0.00551, 0.01555] | 0.00293 | True |

## Reliability and duplicate controls

| Candidate | Split-half rho | Centered rho | Corrected duplicate gain A/B | Same-penalty duplicate gain A/B |
|---|---:|---:|---:|---:|
| 02 | 0.9317 | 0.6648 | 0.00000 / 0.00000 | 0.00001 / 0.00001 |
| 03 | 0.9227 | 0.6818 | 0.00000 / 0.00000 | 0.00000 / 0.00000 |

## Audit boundary

Registration `57e0f9e00da2f5562e6caa2f80e690a9f2fc82f276f139205b6d3d1257f0d112` was sealed before MECHCONF generation. All 128000 confirmation futures were regenerated exactly: **True**.

This supports only the narrow contrasts that passed. It remains a clean-room test of explicit candidate contracts, not an execution of the unavailable original-paper code.
