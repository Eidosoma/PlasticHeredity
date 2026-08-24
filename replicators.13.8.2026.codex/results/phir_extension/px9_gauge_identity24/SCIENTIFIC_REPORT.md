# PX9 high-support gauge-identity pilot

Registration: `2e1aec89d8e6d9e6d51d4257d7886e4b476779359cef1f808f5781e415abda11`.

## Plastic-heredity dose validity

| Test | Candidate | Cell | Effect [95% CI] | Holm p | Pass |
| --- | --- | --- | ---: | ---: | --- |
| renewal Q100-Q00 | 02 | A | +0.0574 [+0.0324, +0.0856] | 0.0009763 | True |
| renewal Q100-Q00 | 02 | B | +0.0529 [+0.0291, +0.0780] | 0.0009763 | True |
| renewal Q100-Q00 | 03 | A | +0.0610 [+0.0379, +0.0874] | 0.0009763 | True |
| renewal Q100-Q00 | 03 | B | +0.0667 [+0.0447, +0.0902] | 0.0009763 | True |
| renewal dose Spearman | 02 | A | +0.2950 [+0.1629, +0.4066] | 0.0009763 | True |
| renewal dose Spearman | 02 | B | +0.3657 [+0.2709, +0.4527] | 0.0009763 | True |
| renewal dose Spearman | 03 | A | +0.3617 [+0.2689, +0.4479] | 0.0009763 | True |
| renewal dose Spearman | 03 | B | +0.3889 [+0.2967, +0.4805] | 0.0009763 | True |

## Temporal authenticity

| Test | Candidate | Cell | Effect [95% CI] | Holm p | Pass |
| --- | --- | --- | ---: | ---: | --- |
| temporal response | 02 | A | +0.0214 [-0.0214, +0.0613] | 0.1706 | False |
| temporal response | 02 | B | +0.0400 [+0.0019, +0.0766] | 0.06444 | False |
| temporal response | 03 | A | +0.0464 [+0.0054, +0.0872] | 0.06444 | False |
| temporal response | 03 | B | +0.0465 [+0.0172, +0.0773] | 0.008787 | True |
| temporal reliability | 02 |  | +0.4402 [+0.3649, +0.5173] | 0.0004882 | True |
| temporal reliability | 03 |  | +0.4090 [+0.3386, +0.4777] | 0.0004882 | True |
| temporal forecast | 02 | A_to_B | +0.2446 [+0.1203, +0.3650] | 0.004393 | True |
| temporal forecast | 02 | B_to_A | +0.2860 [+0.1890, +0.3811] | 0.0009763 | True |
| temporal forecast | 03 | A_to_B | +0.1526 [+0.0460, +0.2520] | 0.004393 | True |
| temporal forecast | 03 | B_to_A | +0.1640 [+0.0700, +0.2571] | 0.004393 | True |
| temporal dose | 02 | A_to_B | +0.0895 [-0.0032, +0.1773] | 0.1338 | False |
| temporal dose | 02 | B_to_A | +0.0647 [-0.0450, +0.1642] | 0.3302 | False |
| temporal dose | 03 | A_to_B | +0.0587 [-0.0425, +0.1573] | 0.3302 | False |
| temporal dose | 03 | B_to_A | +0.0684 [-0.0356, +0.1707] | 0.3302 | False |

## Beta-topology specificity

| Test | Candidate | Cell | Effect [95% CI] | Holm p | Pass |
| --- | --- | --- | ---: | ---: | --- |
| topology response | 02 | A | -0.0117 [-0.0345, +0.0116] | 1 | False |
| topology response | 02 | B | +0.0221 [-0.0025, +0.0477] | 0.2128 | False |
| topology response | 03 | A | +0.0030 [-0.0240, +0.0313] | 1 | False |
| topology response | 03 | B | +0.0035 [-0.0241, +0.0284] | 1 | False |
| topology forecast | 02 | A_to_B | -0.0034 [-0.0850, +0.0849] | 1 | False |
| topology forecast | 02 | B_to_A | +0.1046 [+0.0133, +0.2006] | 0.1015 | False |
| topology forecast | 03 | A_to_B | -0.0707 [-0.1568, +0.0152] | 1 | False |
| topology forecast | 03 | B_to_A | -0.0901 [-0.1980, +0.0103] | 1 | False |

## Behavioral nonredundancy

| Test | Candidate | Direction | Log-loss gain [95% CI] | Holm p | Pass |
| --- | --- | --- | ---: | ---: | --- |
| incremental log loss | 02 | A_to_B | -0.0001 [-0.0002, +0.0001] | 1 | False |
| incremental log loss | 02 | B_to_A | +0.0000 [-0.0002, +0.0002] | 1 | False |
| incremental log loss | 03 | A_to_B | -0.0001 [-0.0001, -0.0000] | 1 | False |
| incremental log loss | 03 | B_to_A | -0.0000 [-0.0001, +0.0001] | 1 | False |

## Raw extension support check

| Source branches | Candidate | Half | Q100-Q00 [95% CI] |
| ---: | --- | --- | ---: |
| 64 | 02 | A | +0.1724 [+0.0845, +0.2712] |
| 64 | 02 | B | +0.1763 [+0.0922, +0.2706] |
| 64 | 03 | A | +0.2240 [+0.1644, +0.2917] |
| 64 | 03 | B | +0.2501 [+0.1805, +0.3367] |
| 128 | 02 | A | +0.0669 [+0.0209, +0.1094] |
| 128 | 02 | B | +0.0945 [+0.0556, +0.1332] |
| 128 | 03 | A | +0.0884 [+0.0543, +0.1246] |
| 128 | 03 | B | +0.0981 [+0.0672, +0.1333] |

## Public revised negative control

| Candidate | Half | Q100-Q00 [95% CI] |
| --- | --- | ---: |
| 02 | A | +0.0171 [-0.0116, +0.0473] |
| 02 | B | +0.0227 [-0.0078, +0.0546] |
| 03 | A | +0.0202 [-0.0087, +0.0493] |
| 03 | B | +0.0053 [-0.0233, +0.0341] |

## Concordant-outcome descriptive diagnostic

| Candidate | Half | Q100-Q00 [95% CI] | Median matched branches |
| --- | --- | ---: | ---: |
| 02 | A | +0.0842 [+0.0287, +0.1371] | 117.0 |
| 02 | B | +0.1205 [+0.0759, +0.1657] | 115.0 |
| 03 | A | +0.1215 [+0.0741, +0.1711] | 113.0 |
| 03 | B | +0.1317 [+0.0931, +0.1735] | 113.0 |

This diagnostic conditions on post-treatment renewal and survival agreement and is not a causal mediation analysis.

## Registered gates and classification

- eligibility: **True**
- plastic_heredity_manipulation_valid: **True**
- temporal_score_complete: **True**
- topology_score_complete: **True**
- temporal_authenticity: **False**
- beta_topology_specificity: **False**
- behaviorally_nonredundant: **False**
- public_revised_positive_all_cells: **False**
- pilot_classification: **finite_sample_or_marginal_explanation**
- automatic_48_matrix_continuation_authorized: **False**

## Claim boundary

PX9 is a prospective 24-matrix mechanistic pilot. It identifies what the PX8 extension behaves like; it cannot rescue the public nine-atom Phi-r, make an information statistic causal, or automatically authorize a 48-matrix continuation.
