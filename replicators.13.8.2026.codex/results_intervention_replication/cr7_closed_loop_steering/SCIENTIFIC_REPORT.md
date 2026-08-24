# CR7 closed-loop hereditary steering

Complete registered 60-fission CR7 gate: **True**.
Exact replay: **True**.
No-op callback/plain identity: **True**.

| Candidate | Contrast | Estimate | 95% CI | Gate |
|---|---|---:|---:|---:|
| 02 | MODEL_DOWN − NOOP inheritance | +0.088426 | (0.06516203703703702, 0.11482204861111109) | True |
| 02 | RULE_DOWN − NOOP inheritance | +0.085590 | (0.06354166666666662, 0.11068431712962962) | True |
| 02 | MODEL_UP − NOOP inheritance | -0.112211 | (-0.13838252314814817, -0.08715277777777779) | True |
| 02 | MODEL_UP − MODEL_DOWN episodes | +2.656250 | (2.222222222222222, 3.0833333333333335) | True |
| 02 | RANDOM − NOOP inheritance | -0.001852 | (-0.010821759259259255, 0.0070601851851851945) | True |
| 02 | RULE_DOWN recovery fraction | +0.967932 | (0.932616939711008, 1.0006727768408004) | strong=True |
| 03 | MODEL_DOWN − NOOP inheritance | +0.095775 | (0.07073929398148152, 0.12291666666666666) | True |
| 03 | RULE_DOWN − NOOP inheritance | +0.092419 | (0.0683449074074074, 0.1187282986111111) | True |
| 03 | MODEL_UP − NOOP inheritance | -0.088773 | (-0.10946903935185184, -0.06935040509259258) | True |
| 03 | MODEL_UP − MODEL_DOWN episodes | +2.770833 | (2.2916666666666665, 3.2591145833333335) | True |
| 03 | RANDOM − NOOP inheritance | -0.002836 | (-0.011031539351851855, 0.005729166666666659) | True |
| 03 | RULE_DOWN recovery fraction | +0.964955 | (0.9342442665515497, 0.9915602784814097) | strong=True |

All confidence intervals use whole catalytic matrices as blocks; candidates were not pooled. Random equivalence uses the complete 90% interval inside +/-0.025, not merely an interval crossing zero.

The conditional second 60-fission active-control extension ran and is reported separately.

This phase tests externally maintained control while interventions continue. It does not test autonomous persistence after release.
