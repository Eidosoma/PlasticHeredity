# Fresh 24-matrix Phi-r window bridge

Registration: `9ef3c307b39d56ee9b82651ee39984ef5c3260cef735243eecd680fb97100c89`. This result is separate from the completed Chapter 5 pilot and its locked confirmation.

## Arm effects: stabilization minus destabilization

| Metric | Candidate | Replicate | Effect [95% matrix CI] | Holm p |
| --- | --- | ---: | ---: | ---: |
| inherited_31_60 | 02 | 0 | +0.2319 [+0.1681, +0.2986] | 0.0009763 |
| inherited_31_60 | 02 | 1 | +0.2806 [+0.2125, +0.3523] | 0.0009763 |
| inherited_31_60 | 03 | 0 | +0.2778 [+0.2033, +0.3550] | 0.0009763 |
| inherited_31_60 | 03 | 1 | +0.2389 [+0.1653, +0.3167] | 0.0009763 |
| pooled20_clr_revised | 02 | 0 | -0.1930 [-0.3697, -0.0184] | 1 |
| pooled20_clr_revised | 02 | 1 | -0.2228 [-0.4032, -0.0556] | 1 |
| pooled20_clr_revised | 03 | 0 | -0.4001 [-0.5502, -0.2583] | 1 |
| pooled20_clr_revised | 03 | 1 | -0.1954 [-0.4007, -0.0064] | 1 |
| rolling20_clr_revised | 02 | 0 | -0.1237 [-0.2036, -0.0459] | 1 |
| rolling20_clr_revised | 02 | 1 | -0.1358 [-0.2214, -0.0501] | 1 |
| rolling20_clr_revised | 03 | 0 | -0.2569 [-0.3503, -0.1684] | 1 |
| rolling20_clr_revised | 03 | 1 | -0.0919 [-0.1980, +0.0027] | 1 |
| pooled30_clr_revised | 02 | 0 | -0.2992 [-0.4180, -0.1886] | 1 |
| pooled30_clr_revised | 02 | 1 | -0.1952 [-0.3737, -0.0192] | 1 |
| pooled30_clr_revised | 03 | 0 | -0.3787 [-0.5336, -0.2181] | 1 |
| pooled30_clr_revised | 03 | 1 | -0.1917 [-0.3337, -0.0523] | 1 |
| rolling30_clr_revised | 02 | 0 | -0.1099 [-0.1780, -0.0468] | 1 |
| rolling30_clr_revised | 02 | 1 | -0.1349 [-0.2023, -0.0657] | 1 |
| rolling30_clr_revised | 03 | 0 | -0.2507 [-0.3291, -0.1770] | 1 |
| rolling30_clr_revised | 03 | 1 | -0.0972 [-0.1893, -0.0116] | 1 |

## Paired estimator moderation: rolling minus pooled arm effect

| Range | Candidate | Replicate | Effect [95% matrix CI] | Holm p |
| --- | --- | ---: | ---: | ---: |
| moderation20 | 02 | 0 | +0.0693 [-0.0550, +0.1916] | 1 |
| moderation20 | 02 | 1 | +0.0871 [-0.0662, +0.2512] | 1 |
| moderation20 | 03 | 0 | +0.1431 [+0.0333, +0.2551] | 1 |
| moderation20 | 03 | 1 | +0.1036 [-0.0570, +0.2643] | 1 |
| moderation30 | 02 | 0 | +0.1892 [+0.0857, +0.2954] | 1 |
| moderation30 | 02 | 1 | +0.0603 [-0.0990, +0.2128] | 1 |
| moderation30 | 03 | 0 | +0.1280 [-0.0087, +0.2647] | 1 |
| moderation30 | 03 | 1 | +0.0945 [-0.0396, +0.2370] | 1 |

## Registered gates

- heredity_validity: **True**
- pooled20_response: **False**
- moderation20: **False**
- moderation30: **False**
- full_sign_reversal: **False**

## Registered atom, preprocessing, and typeset sensitivities

The full-dimensional rolling typeset value uses only boundaries 40 and 60, matching the completed Codex pilot; macro-typeset and revised readings use every registered rolling boundary.

| Metric | Candidate | Replicate | Effect [95% matrix CI] |
| --- | --- | ---: | ---: |
| pooled20_clr_full_typeset | 02 | 0 | +3.1198 [+0.5558, +5.8883] |
| pooled20_clr_full_typeset | 02 | 1 | +4.4487 [+1.6427, +7.2277] |
| pooled20_clr_full_typeset | 03 | 0 | +3.3672 [+0.4760, +6.1683] |
| pooled20_clr_full_typeset | 03 | 1 | +3.6852 [+0.6157, +6.8142] |
| pooled20_clr_macro_typeset | 02 | 0 | +0.0552 [-0.0006, +0.1185] |
| pooled20_clr_macro_typeset | 02 | 1 | +0.0373 [-0.0162, +0.0909] |
| pooled20_clr_macro_typeset | 03 | 0 | +0.0189 [-0.0250, +0.0643] |
| pooled20_clr_macro_typeset | 03 | 1 | +0.0333 [-0.0221, +0.0856] |
| pooled20_clr_causation | 02 | 0 | +0.0749 [-0.3353, +0.4792] |
| pooled20_clr_causation | 02 | 1 | -0.0799 [-0.4110, +0.2410] |
| pooled20_clr_causation | 03 | 0 | +0.2951 [-0.1006, +0.6718] |
| pooled20_clr_causation | 03 | 1 | -0.3487 [-0.6631, -0.0311] |
| pooled20_clr_emergence | 02 | 0 | -0.1283 [-0.2273, -0.0268] |
| pooled20_clr_emergence | 02 | 1 | -0.1674 [-0.2518, -0.0857] |
| pooled20_clr_emergence | 03 | 0 | -0.1777 [-0.3115, -0.0574] |
| pooled20_clr_emergence | 03 | 1 | -0.2054 [-0.2916, -0.1327] |
| pooled20_clr_synergy | 02 | 0 | -0.2032 [-0.5848, +0.1576] |
| pooled20_clr_synergy | 02 | 1 | -0.0875 [-0.3950, +0.2220] |
| pooled20_clr_synergy | 03 | 0 | -0.4728 [-0.7655, -0.1736] |
| pooled20_clr_synergy | 03 | 1 | +0.1433 [-0.1700, +0.4524] |
| pooled20_raw_count_revised | 02 | 0 | -0.2407 [-0.3567, -0.1117] |
| pooled20_raw_count_revised | 02 | 1 | -0.3269 [-0.4123, -0.2477] |
| pooled20_raw_count_revised | 03 | 0 | -0.2687 [-0.3785, -0.1679] |
| pooled20_raw_count_revised | 03 | 1 | -0.2758 [-0.3942, -0.1609] |
| rolling30_clr_full_typeset | 02 | 0 | +0.7097 [-0.8110, +2.2147] |
| rolling30_clr_full_typeset | 02 | 1 | +1.7897 [-0.1080, +3.7045] |
| rolling30_clr_full_typeset | 03 | 0 | +2.0752 [-0.4994, +4.7429] |
| rolling30_clr_full_typeset | 03 | 1 | +2.7481 [+0.4744, +5.1596] |
| rolling30_clr_macro_typeset | 02 | 0 | +0.0336 [+0.0098, +0.0586] |
| rolling30_clr_macro_typeset | 02 | 1 | +0.0039 [-0.0228, +0.0290] |
| rolling30_clr_macro_typeset | 03 | 0 | +0.0291 [+0.0022, +0.0546] |
| rolling30_clr_macro_typeset | 03 | 1 | +0.0498 [+0.0275, +0.0743] |
| rolling30_clr_causation | 02 | 0 | -0.0282 [-0.1912, +0.1323] |
| rolling30_clr_causation | 02 | 1 | +0.0541 [-0.1525, +0.2457] |
| rolling30_clr_causation | 03 | 0 | +0.0281 [-0.1751, +0.2306] |
| rolling30_clr_causation | 03 | 1 | -0.0993 [-0.3152, +0.0999] |
| rolling30_clr_emergence | 02 | 0 | -0.0855 [-0.1491, -0.0220] |
| rolling30_clr_emergence | 02 | 1 | -0.0960 [-0.1736, -0.0211] |
| rolling30_clr_emergence | 03 | 0 | -0.1431 [-0.2122, -0.0738] |
| rolling30_clr_emergence | 03 | 1 | -0.0866 [-0.1637, -0.0097] |
| rolling30_clr_synergy | 02 | 0 | -0.0573 [-0.1740, +0.0656] |
| rolling30_clr_synergy | 02 | 1 | -0.1501 [-0.2898, +0.0078] |
| rolling30_clr_synergy | 03 | 0 | -0.1712 [-0.3243, -0.0156] |
| rolling30_clr_synergy | 03 | 1 | +0.0127 [-0.1461, +0.1818] |
| rolling30_raw_count_revised | 02 | 0 | +0.1049 [-0.2161, +0.4232] |
| rolling30_raw_count_revised | 02 | 1 | +0.2821 [-0.1595, +0.7244] |
| rolling30_raw_count_revised | 03 | 0 | +0.2247 [-0.1294, +0.5872] |
| rolling30_raw_count_revised | 03 | 1 | +0.0512 [-0.2655, +0.3562] |

All 16 individual atom effects are retained in `primary_metrics.json` and `matrix_effects.csv.gz`.

## Partition reconfiguration

Zero is an identical bipartition up to swapping its labels; 0.5 is maximal disagreement on common active coordinates.

| Preprocessing | Pooled reference | Candidate | Replicate | Arm | Mean disagreement | Windows |
| --- | --- | --- | ---: | --- | ---: | ---: |
| clr | pooled20 | 02 | 0 | MODEL_DESTABILIZE | 0.265 | 720 |
| clr | pooled20 | 02 | 0 | MODEL_STABILIZE | 0.236 | 720 |
| clr | pooled20 | 02 | 1 | MODEL_DESTABILIZE | 0.295 | 720 |
| clr | pooled20 | 02 | 1 | MODEL_STABILIZE | 0.216 | 720 |
| clr | pooled20 | 03 | 0 | MODEL_DESTABILIZE | 0.276 | 720 |
| clr | pooled20 | 03 | 0 | MODEL_STABILIZE | 0.219 | 720 |
| clr | pooled20 | 03 | 1 | MODEL_DESTABILIZE | 0.282 | 720 |
| clr | pooled20 | 03 | 1 | MODEL_STABILIZE | 0.250 | 720 |
| clr | pooled30 | 02 | 0 | MODEL_DESTABILIZE | 0.255 | 720 |
| clr | pooled30 | 02 | 0 | MODEL_STABILIZE | 0.228 | 720 |
| clr | pooled30 | 02 | 1 | MODEL_DESTABILIZE | 0.266 | 720 |
| clr | pooled30 | 02 | 1 | MODEL_STABILIZE | 0.226 | 720 |
| clr | pooled30 | 03 | 0 | MODEL_DESTABILIZE | 0.274 | 720 |
| clr | pooled30 | 03 | 0 | MODEL_STABILIZE | 0.246 | 720 |
| clr | pooled30 | 03 | 1 | MODEL_DESTABILIZE | 0.269 | 720 |
| clr | pooled30 | 03 | 1 | MODEL_STABILIZE | 0.245 | 720 |
| raw_count | pooled20 | 02 | 0 | MODEL_DESTABILIZE | 0.067 | 720 |
| raw_count | pooled20 | 02 | 0 | MODEL_STABILIZE | 0.092 | 720 |
| raw_count | pooled20 | 02 | 1 | MODEL_DESTABILIZE | 0.075 | 720 |
| raw_count | pooled20 | 02 | 1 | MODEL_STABILIZE | 0.075 | 720 |
| raw_count | pooled20 | 03 | 0 | MODEL_DESTABILIZE | 0.097 | 720 |
| raw_count | pooled20 | 03 | 0 | MODEL_STABILIZE | 0.087 | 720 |
| raw_count | pooled20 | 03 | 1 | MODEL_DESTABILIZE | 0.083 | 690 |
| raw_count | pooled20 | 03 | 1 | MODEL_STABILIZE | 0.101 | 720 |
| raw_count | pooled30 | 02 | 0 | MODEL_DESTABILIZE | 0.079 | 720 |
| raw_count | pooled30 | 02 | 0 | MODEL_STABILIZE | 0.086 | 720 |
| raw_count | pooled30 | 02 | 1 | MODEL_DESTABILIZE | 0.071 | 720 |
| raw_count | pooled30 | 02 | 1 | MODEL_STABILIZE | 0.079 | 720 |
| raw_count | pooled30 | 03 | 0 | MODEL_DESTABILIZE | 0.097 | 690 |
| raw_count | pooled30 | 03 | 0 | MODEL_STABILIZE | 0.080 | 720 |
| raw_count | pooled30 | 03 | 1 | MODEL_DESTABILIZE | 0.102 | 720 |
| raw_count | pooled30 | 03 | 1 | MODEL_STABILIZE | 0.099 | 720 |

## Completion and eligibility

| Candidate | Replicate | Arm | Completed | Information eligible |
| --- | ---: | --- | ---: | ---: |
| 02 | 0 | MODEL_DESTABILIZE | 24/24 | 24/24 |
| 02 | 0 | MODEL_STABILIZE | 24/24 | 24/24 |
| 02 | 1 | MODEL_DESTABILIZE | 24/24 | 24/24 |
| 02 | 1 | MODEL_STABILIZE | 24/24 | 24/24 |
| 03 | 0 | MODEL_DESTABILIZE | 24/24 | 24/24 |
| 03 | 0 | MODEL_STABILIZE | 24/24 | 24/24 |
| 03 | 1 | MODEL_DESTABILIZE | 24/24 | 24/24 |
| 03 | 1 | MODEL_STABILIZE | 24/24 | 24/24 |

## Boundaries

The result tests temporal estimator dependence inside the two Codex GARD contracts. It does not select a uniquely correct Phi-r, make Phi-r a controller, or support consciousness, life, agency, biological memory, a universal origin-of-life mechanism, or a Platonic-space portal.
