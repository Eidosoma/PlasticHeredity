# CR10 exploratory internalization ladder

CR10 is exploratory and has no confirmatory pass/fail gate. It cannot rescue or replace CR6, CR8, CR9, or any other sealed result.

## Home-regime maintenance and K8 recovery

All entries below are policy-minus-NOOP inherited-boundary effects with whole-matrix 95% bootstrap intervals.

| Candidate | Condition | Policy | Effect | 95% CI | Holm p |
|---|---|---|---:|---:|---:|
| 02 | UNCHALLENGED | L0_RULE_CONTINUOUS | +0.0899 | [+0.0647, +0.1168] | 0.002929 |
| 02 | UNCHALLENGED | L1_RULE_AFTER_BREAK | +0.0326 | [+0.0204, +0.0469] | 0.002929 |
| 02 | UNCHALLENGED | L2_RULE_UNTIL_RUN3 | +0.0403 | [+0.0250, +0.0580] | 0.002929 |
| 02 | UNCHALLENGED | L3_LOCAL_TREE | +0.0905 | [+0.0649, +0.1180] | 0.002929 |
| 02 | UNCHALLENGED | MODEL_DOWN | +0.0924 | [+0.0664, +0.1199] | 0.002929 |
| 02 | UNCHALLENGED | RANDOM | -0.0155 | [-0.0270, -0.0044] | 0.0249 |
| 02 | CHALLENGED_K8 | L0_RULE_CONTINUOUS | +0.1030 | [+0.0769, +0.1322] | 0.002929 |
| 02 | CHALLENGED_K8 | L1_RULE_AFTER_BREAK | +0.0275 | [+0.0128, +0.0433] | 0.002929 |
| 02 | CHALLENGED_K8 | L2_RULE_UNTIL_RUN3 | +0.0479 | [+0.0303, +0.0678] | 0.002929 |
| 02 | CHALLENGED_K8 | L3_LOCAL_TREE | +0.1028 | [+0.0762, +0.1312] | 0.002929 |
| 02 | CHALLENGED_K8 | MODEL_DOWN | +0.1060 | [+0.0799, +0.1347] | 0.002929 |
| 02 | CHALLENGED_K8 | RANDOM | -0.0072 | [-0.0240, +0.0093] | 0.4098 |
| 03 | UNCHALLENGED | L0_RULE_CONTINUOUS | +0.1017 | [+0.0752, +0.1305] | 0.002929 |
| 03 | UNCHALLENGED | L1_RULE_AFTER_BREAK | +0.0308 | [+0.0183, +0.0446] | 0.002929 |
| 03 | UNCHALLENGED | L2_RULE_UNTIL_RUN3 | +0.0495 | [+0.0329, +0.0675] | 0.002929 |
| 03 | UNCHALLENGED | L3_LOCAL_TREE | +0.1044 | [+0.0775, +0.1331] | 0.002929 |
| 03 | UNCHALLENGED | MODEL_DOWN | +0.1025 | [+0.0756, +0.1321] | 0.002929 |
| 03 | UNCHALLENGED | RANDOM | +0.0008 | [-0.0126, +0.0169] | 0.9236 |
| 03 | CHALLENGED_K8 | L0_RULE_CONTINUOUS | +0.1081 | [+0.0819, +0.1359] | 0.002929 |
| 03 | CHALLENGED_K8 | L1_RULE_AFTER_BREAK | +0.0370 | [+0.0194, +0.0557] | 0.002929 |
| 03 | CHALLENGED_K8 | L2_RULE_UNTIL_RUN3 | +0.0542 | [+0.0354, +0.0745] | 0.002929 |
| 03 | CHALLENGED_K8 | L3_LOCAL_TREE | +0.1088 | [+0.0824, +0.1370] | 0.002929 |
| 03 | CHALLENGED_K8 | MODEL_DOWN | +0.1060 | [+0.0799, +0.1338] | 0.002929 |
| 03 | CHALLENGED_K8 | RANDOM | -0.0137 | [-0.0326, +0.0051] | 0.3505 |

The challenged analysis uses the registered fissions 31--60 window. Challenge-minus-unchallenged effects and local-policy fractions of the MODEL_DOWN gain are retained in `inference_metrics.json`.

## Zero-shot transfer

| Regime | Candidate | Policy | Effect vs NOOP | 95% CI |
|---|---|---|---:|---:|
| POS_A_M4_S5 | 02 | L0_RULE_CONTINUOUS | +0.0198 | [+0.0108, +0.0299] |
| POS_A_M4_S5 | 02 | L1_RULE_AFTER_BREAK | +0.0003 | [-0.0024, +0.0042] |
| POS_A_M4_S5 | 02 | L2_RULE_UNTIL_RUN3 | +0.0073 | [+0.0014, +0.0142] |
| POS_A_M4_S5 | 02 | L3_LOCAL_TREE | +0.0212 | [+0.0115, +0.0319] |
| POS_A_M4_S5 | 02 | MODEL_DOWN | +0.0205 | [+0.0111, +0.0311] |
| POS_A_M4_S5 | 02 | RANDOM | -0.0059 | [-0.0122, -0.0003] |
| POS_A_M4_S5 | 03 | L0_RULE_CONTINUOUS | +0.0354 | [+0.0212, +0.0514] |
| POS_A_M4_S5 | 03 | L1_RULE_AFTER_BREAK | +0.0080 | [-0.0002, +0.0167] |
| POS_A_M4_S5 | 03 | L2_RULE_UNTIL_RUN3 | +0.0142 | [+0.0066, +0.0229] |
| POS_A_M4_S5 | 03 | L3_LOCAL_TREE | +0.0375 | [+0.0229, +0.0537] |
| POS_A_M4_S5 | 03 | MODEL_DOWN | +0.0344 | [+0.0198, +0.0507] |
| POS_A_M4_S5 | 03 | RANDOM | -0.0076 | [-0.0188, +0.0035] |
| POS_A_M3_S4 | 02 | L0_RULE_CONTINUOUS | +0.0743 | [+0.0444, +0.1104] |
| POS_A_M3_S4 | 02 | L1_RULE_AFTER_BREAK | +0.0222 | [+0.0062, +0.0434] |
| POS_A_M3_S4 | 02 | L2_RULE_UNTIL_RUN3 | +0.0337 | [+0.0146, +0.0596] |
| POS_A_M3_S4 | 02 | L3_LOCAL_TREE | +0.0753 | [+0.0462, +0.1104] |
| POS_A_M3_S4 | 02 | MODEL_DOWN | +0.0726 | [+0.0438, +0.1080] |
| POS_A_M3_S4 | 02 | RANDOM | -0.0007 | [-0.0139, +0.0115] |
| POS_A_M3_S4 | 03 | L0_RULE_CONTINUOUS | +0.0747 | [+0.0465, +0.1083] |
| POS_A_M3_S4 | 03 | L1_RULE_AFTER_BREAK | +0.0181 | [+0.0007, +0.0372] |
| POS_A_M3_S4 | 03 | L2_RULE_UNTIL_RUN3 | +0.0292 | [+0.0076, +0.0549] |
| POS_A_M3_S4 | 03 | L3_LOCAL_TREE | +0.0747 | [+0.0472, +0.1076] |
| POS_A_M3_S4 | 03 | MODEL_DOWN | +0.0750 | [+0.0458, +0.1094] |
| POS_A_M3_S4 | 03 | RANDOM | -0.0240 | [-0.0389, -0.0101] |
| POS_A_M5_S4 | 02 | L0_RULE_CONTINUOUS | +0.1674 | [+0.1139, +0.2226] |
| POS_A_M5_S4 | 02 | L1_RULE_AFTER_BREAK | +0.0674 | [+0.0427, +0.0941] |
| POS_A_M5_S4 | 02 | L2_RULE_UNTIL_RUN3 | +0.0865 | [+0.0531, +0.1222] |
| POS_A_M5_S4 | 02 | L3_LOCAL_TREE | +0.1674 | [+0.1170, +0.2181] |
| POS_A_M5_S4 | 02 | MODEL_DOWN | +0.1660 | [+0.1135, +0.2204] |
| POS_A_M5_S4 | 02 | RANDOM | +0.0014 | [-0.0287, +0.0326] |
| POS_A_M5_S4 | 03 | L0_RULE_CONTINUOUS | +0.1510 | [+0.1059, +0.2010] |
| POS_A_M5_S4 | 03 | L1_RULE_AFTER_BREAK | +0.0462 | [+0.0187, +0.0764] |
| POS_A_M5_S4 | 03 | L2_RULE_UNTIL_RUN3 | +0.0733 | [+0.0392, +0.1104] |
| POS_A_M5_S4 | 03 | L3_LOCAL_TREE | +0.1503 | [+0.1035, +0.2020] |
| POS_A_M5_S4 | 03 | MODEL_DOWN | +0.1524 | [+0.1035, +0.2052] |
| POS_A_M5_S4 | 03 | RANDOM | -0.0247 | [-0.0513, -0.0014] |

## Retention-only kinetic prototype

| Candidate | Lambda | Inheritance change vs 0 | 95% CI |
|---|---:|---:|---:|
| 02 | 0.1 | -0.0094 | [-0.0212, +0.0028] |
| 02 | 0.3 | +0.0046 | [-0.0084, +0.0168] |
| 03 | 0.1 | -0.0039 | [-0.0178, +0.0103] |
| 03 | 0.3 | +0.0023 | [-0.0088, +0.0152] |

## Integrity and claim boundary

Policy replay exact: **True**. Kinetic replay exact: **True**. NOOP/plain identity: **True**. Lambda-zero/plain identity: **True**.

These results concern externally applied local policies and one retention-only model extension. They do not demonstrate autonomous agency, biological memory, installed compotypes, life, real prebiotic chemistry, strict-eight control, or a universal origin-of-life mechanism.
