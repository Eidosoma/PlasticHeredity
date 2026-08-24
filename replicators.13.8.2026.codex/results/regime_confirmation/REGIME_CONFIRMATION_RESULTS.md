# Prospective coherent-regime confirmation

## Outcome

The primary occurrence gate passed: the simulator prospectively produced distinct, mutually coherent, persistent eight-fission hereditary regimes in both candidates and both branch halves.
The frozen state-added predictor did not pass all four prospective gates.

## Endpoint occurrence

| Endpoint | Candidate | Half | Rate | 95% matrix-bootstrap CI | Events | Event matrices |
|---|---:|---:|---:|---:|---:|---:|
| primary_all8 | 02 | A | 0.018687 | [0.013912, 0.023688] | 1196 | 119 |
| primary_all8 | 02 | B | 0.018094 | [0.013865, 0.022828] | 1158 | 126 |
| primary_all8 | 03 | A | 0.020891 | [0.015631, 0.026484] | 1337 | 129 |
| primary_all8 | 03 | B | 0.021094 | [0.016256, 0.026766] | 1350 | 143 |
| secondary_first5 | 02 | A | 0.071687 | [0.061230, 0.082369] | 4588 | 181 |
| secondary_first5 | 02 | B | 0.071375 | [0.061334, 0.081822] | 4568 | 179 |
| secondary_first5 | 03 | A | 0.080422 | [0.069068, 0.092385] | 5147 | 177 |
| secondary_first5 | 03 | B | 0.083469 | [0.071652, 0.095213] | 5342 | 181 |
| secondary_centroid | 02 | A | 0.082688 | [0.073365, 0.093020] | 5292 | 188 |
| secondary_centroid | 02 | B | 0.082656 | [0.072828, 0.092773] | 5290 | 190 |
| secondary_centroid | 03 | A | 0.094125 | [0.082871, 0.105885] | 6024 | 190 |
| secondary_centroid | 03 | B | 0.093469 | [0.082365, 0.105113] | 5982 | 191 |

## Primary state-added prediction tests

| Candidate | Half | Log-loss gain | 95% CI | Holm p | Statistical pass |
|---|---:|---:|---:|---:|---:|
| 02 | A | 0.000525 | [-0.000064, 0.001229] | 0.119112 | False |
| 02 | B | 0.000284 | [-0.000219, 0.000843] | 0.155236 | False |
| 03 | A | 0.000388 | [0.000084, 0.000706] | 0.035148 | True |
| 03 | B | 0.000356 | [0.000027, 0.000684] | 0.051989 | False |

## Prespecified secondary state-added contrasts

| Endpoint | Candidate | Half | Log-loss gain | 95% CI |
|---|---:|---:|---:|---:|
| secondary_first5 | 02 | A | 0.000898 | [-0.000000, 0.001828] |
| secondary_first5 | 02 | B | 0.000622 | [-0.000261, 0.001577] |
| secondary_first5 | 03 | A | 0.001517 | [0.000455, 0.002633] |
| secondary_first5 | 03 | B | 0.001401 | [0.000155, 0.002614] |
| secondary_centroid | 02 | A | 0.000889 | [-0.000193, 0.001995] |
| secondary_centroid | 02 | B | 0.000749 | [-0.000330, 0.001836] |
| secondary_centroid | 03 | A | 0.001236 | [0.000089, 0.002428] |
| secondary_centroid | 03 | B | 0.001035 | [-0.000260, 0.002334] |

## Descriptive frozen-model ranks

| Endpoint | Candidate | Model | Overall Spearman mean | Within-matrix Spearman mean |
|---|---:|---|---:|---:|
| primary_all8 | 02 | h10 | 0.242208 | 0.061845 |
| primary_all8 | 02 | h10_state | 0.283654 | 0.029257 |
| primary_all8 | 02 | h10_state_beta | 0.356341 | 0.031283 |
| primary_all8 | 02 | h10_state_beta_interaction | 0.361114 | 0.033403 |
| primary_all8 | 02 | state_only | 0.152824 | -0.015720 |
| primary_all8 | 02 | beta_only | 0.246889 | -0.012132 |
| primary_all8 | 02 | h10_beta | 0.327449 | 0.063864 |
| primary_all8 | 03 | h10 | 0.152156 | 0.071635 |
| primary_all8 | 03 | h10_state | 0.191080 | 0.035793 |
| primary_all8 | 03 | h10_state_beta | 0.196039 | 0.035843 |
| primary_all8 | 03 | h10_state_beta_interaction | 0.244801 | 0.050403 |
| primary_all8 | 03 | state_only | 0.128026 | 0.004005 |
| primary_all8 | 03 | beta_only | 0.354910 | nan |
| primary_all8 | 03 | h10_beta | 0.157315 | 0.071800 |
| secondary_first5 | 02 | h10 | 0.355124 | 0.202258 |
| secondary_first5 | 02 | h10_state | 0.382260 | 0.189358 |
| secondary_first5 | 02 | h10_state_beta | 0.478415 | 0.188027 |
| secondary_first5 | 02 | h10_state_beta_interaction | 0.482819 | 0.193931 |
| secondary_first5 | 02 | state_only | 0.138065 | 0.060626 |
| secondary_first5 | 02 | beta_only | 0.342865 | -0.023588 |
| secondary_first5 | 02 | h10_beta | 0.451152 | 0.199867 |
| secondary_first5 | 03 | h10 | 0.325366 | 0.149859 |
| secondary_first5 | 03 | h10_state | 0.366782 | 0.142198 |
| secondary_first5 | 03 | h10_state_beta | 0.491894 | 0.139195 |
| secondary_first5 | 03 | h10_state_beta_interaction | 0.499220 | 0.143707 |
| secondary_first5 | 03 | state_only | 0.183850 | 0.100757 |
| secondary_first5 | 03 | beta_only | 0.348697 | -0.002353 |
| secondary_first5 | 03 | h10_beta | 0.446212 | 0.148030 |
| secondary_centroid | 02 | h10 | 0.338279 | 0.228386 |
| secondary_centroid | 02 | h10_state | 0.360075 | 0.233771 |
| secondary_centroid | 02 | h10_state_beta | 0.367893 | 0.235274 |
| secondary_centroid | 02 | h10_state_beta_interaction | 0.378303 | 0.240911 |
| secondary_centroid | 02 | state_only | 0.161293 | 0.219570 |
| secondary_centroid | 02 | beta_only | 0.119021 | -0.009596 |
| secondary_centroid | 02 | h10_beta | 0.352514 | 0.230626 |
| secondary_centroid | 03 | h10 | 0.270828 | 0.163119 |
| secondary_centroid | 03 | h10_state | 0.288042 | 0.170538 |
| secondary_centroid | 03 | h10_state_beta | 0.297079 | 0.170721 |
| secondary_centroid | 03 | h10_state_beta_interaction | 0.305838 | 0.174462 |
| secondary_centroid | 03 | state_only | 0.139184 | 0.188855 |
| secondary_centroid | 03 | beta_only | 0.055029 | 0.010338 |
| secondary_centroid | 03 | h10_beta | 0.284286 | 0.164124 |

## Audit and boundary

Design registration: `71dd86609d52e9853afee34d304dcdfd1afaa7556e51c51e9b6ce9b73561b277`. Model seal: `23d91ed55363de348ad4fd8fccc200e35822642eacd3e830e4d05b28bde36b4a`.
All discrete futures replayed exactly: **True**. Maximum continuous replay error: `0`.

The first-five and centroid endpoints were prespecified secondary analyses and cannot replace the all-eight pairwise primary endpoint. This campaign does not test recurrence, attractor switching, perturbation recovery, causality, biological memory, or prebiotic realism.
