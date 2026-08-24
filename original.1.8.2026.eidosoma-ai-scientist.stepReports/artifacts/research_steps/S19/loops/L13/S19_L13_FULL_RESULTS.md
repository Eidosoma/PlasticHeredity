# S19-L13 Full Results — Figure-5 Recurring-Target Prediction Reconstruction

## Top summary

- **Research step ID:** `S19-L13` (`E01-S19-L13-FIGURE5-RECURRING-TARGET-PREDICTION-RECONSTRUCTION-v1.0.0`)
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Artifacts written:** 49 required named artifacts, 14 figures, exact target/feature/model evidence, validation manifests, and append-only S19 handoff records under `/artifacts/research_steps/S19/loops/L13`.
- **Validation result:** PASS: 400/400 target replays, 2400/2400 feature replays, 28 registered actual-model replay rows, and 2000 immutable prior files with 0 mismatches; 2 value-preserving technical amendments recorded.
- **Outcome classification:** `FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED`; promotion status `NOT_PROMOTABLE`.
- **Caveats or blockers:** This is an adaptive forensic reconstruction on previously studied matrices. R1 and U2 are completed-run recurring-attractor targets. P1 and the oracle are future-dependent. L12 froze approximate Figure 5 centers but not numerical box/whisker endpoints, so the prospectively fixed ±0.05 envelope is an adjudication tolerance rather than a redigitized whisker claim. Two value-preserving technical amendments are fully disclosed; neither changed a scientific value. No result identifies author code or changes S18.
- **Recommended next action:** Mandatory human review. Keep S20 and E02 inactive; do not run confirmation, another loop, intervention, or report generation automatically.

## Lay summary

The paper's Figure 5 shows a dummy predictor near 60% accuracy, while the adjacent-similarity label previously used in S16 made almost every state positive and therefore gave a dummy near 98%. L13 changed only that prediction target. It tested two previously frozen definitions of membership in recurring composition-space attractors while preserving the exact S16 tensor layout, splits, neural network, and training rules. The result below separates the arithmetic baseline clue, a completed-trajectory retrospective reconstruction, and genuinely prefix-only prediction. A completed-fit resemblance cannot be called early warning because its first-quarter feature was fitted using the full trajectory.

## Frozen question and interpretation boundary

The sole question was whether exact L10 R1 or authoritative L11R U2 labels could explain the Figure 5 approximately 60% dummy and model ordering. No third target, new threshold, new clustering rule, simulator change, balancing rule, architecture change, or hyperparameter search was allowed. The Table 1 88% target question was not reopened. A positive retrospective result would imply that Figure 5 likely used a different target object or denominator; it would not prove prospective prediction, causal emergence, intervention efficacy, or author-code identity.

## Inputs and provenance

- Frozen L11R dataset: exactly 100 shared catalytic matrices, 100 candidate-2 and 100 candidate-3 original-exposure trajectories, each with 100 fissions.
- R1 implementation: frozen L10 MATLAB-compatible historical dominant-compotype pipeline.
- U2 implementation: authoritative L11R repaired Euclidean recurring-centroid-union pipeline.
- Prediction implementation: frozen S16 original-order, right-padded, explicitly masked MLP contract and ten matrix-level 64/16/20 fit/validation/test assignments.
- PhiRL: pinned source commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`, source-defined `emergence = synergy + downward causation`, CPU float64.
- Historical GARD: pinned commit `86dff6320d5ae91b4e831471079ff46749b14df9`.
- Paper constraint: exact L12 stored approximate centers (PhiRL 0.85, composition change 0.80, raw 0.80, flux 0.79, dummy 0.60); L12 did not store numeric whisker endpoints.

No mounted external dataset was used. All scientific inputs were frozen E01 artifacts and repository code.

## Detailed methods

### Outcome-blind lock and fixtures

The complete target, tensor, split, feature, model, metric, control, gate, promotion-priority, and resource contract was committed and pushed before L13 target geometry was opened. Sixteen fixtures checked R1 eligibility/ineligibility/ties, U2 exact replay and centroid tolerance, S16 cutoff/masking/splits/dummy/scaling, completed-fit dependence, prefix-only suffix invariance, target-feature separation, typed serialization/quarantine, exact model replay, and worker-failure provenance.

### Target geometry

Each direct molecular target was created on its original selected molecular clock. Undefined R1 trajectories retained a false target mask and were neither imputed nor replaced. The majority probability came only from valid fit-partition labels. The advancement gate required at least 80 defined matrices per candidate, both test classes, no padding or undefined-row scoring, and a ten-split dummy distribution whose range overlapped and median lay inside [0.55, 0.65] in both candidates.

### Features and models

For targets passing geometry, the same feature tensors were reused for both labels: completed-fit PhiRL (P1), prefix-only PhiRL (P2), composition change, raw counts, flux, adjacent H, prefix-only historical attractor geometry, time, matched random values, and P2 combinations with H and prefix geometry. The completed target-centroid oracle was diagnostic only. Every learned family used the identical 288,789-parameter S16 CPU-float64 MLP, AdamW, loss, early stopping, and model seed. Padding never entered loss or metrics.

### Controls and statistics

Controls were within-prefix temporal permutation, training-matrix suffix-label permutation, time only, matched random features, deterministic suffix perturbation, and the excluded completed-centroid oracle. Accuracy was paper-facing; balanced accuracy, AUROC, AUPRC, Brier, log loss, sensitivity, specificity, predictive values, calibration intercept/slope and ECE were secondary. Mann–Whitney reproduced the paper-like ten-split diagnostic; paired Wilcoxon and 4,096-replicate catalytic-matrix bootstraps were the stronger dependence-aware analyses. Candidates and targets were never pooled to rescue a gate.

## Results

### Availability, target geometry, and dummy baseline

| targetId                                   | candidateId   |   definedMatrices |
|:-------------------------------------------|:--------------|------------------:|
| F5_R1_HISTORICAL_DOMINANT_COMPTYPE_H090    | CANDIDATE_2   |                89 |
| F5_R1_HISTORICAL_DOMINANT_COMPTYPE_H090    | CANDIDATE_3   |                86 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   |               100 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   |               100 |

| targetId                                   | candidateId   |   wholeOccupancy |   suffixOccupancy |   rawFirstOnset |   preOnsetAtCutoff |   futureOnset |
|:-------------------------------------------|:--------------|-----------------:|------------------:|----------------:|-------------------:|--------------:|
| F5_R1_HISTORICAL_DOMINANT_COMPTYPE_H090    | CANDIDATE_2   |           0.3774 |            0.4028 |        101.6292 |                 12 |            12 |
| F5_R1_HISTORICAL_DOMINANT_COMPTYPE_H090    | CANDIDATE_3   |           0.3827 |            0.4013 |         78.6628 |                  6 |             6 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   |           0.3978 |            0.4170 |         47.0700 |                  3 |             3 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   |           0.4092 |            0.4258 |         60.7500 |                  5 |             5 |

| targetId                                   | candidateId   |   dummyMedian |   dummyMin |   dummyMax |
|:-------------------------------------------|:--------------|--------------:|-----------:|-----------:|
| F5_R1_HISTORICAL_DOMINANT_COMPTYPE_H090    | CANDIDATE_2   |        0.6557 |     0.5628 |     0.7733 |
| F5_R1_HISTORICAL_DOMINANT_COMPTYPE_H090    | CANDIDATE_3   |        0.6628 |     0.5478 |     0.7496 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   |        0.6425 |     0.5616 |     0.7352 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   |        0.6020 |     0.5541 |     0.7277 |

### Prediction results

| targetId                                   | candidateId   | modelId                               |   medianAccuracy |   medianBalancedAccuracy |   medianAUROC |   medianAUPRC |   medianBrier |
|:-------------------------------------------|:--------------|:--------------------------------------|-----------------:|-------------------------:|--------------:|--------------:|--------------:|
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | B1_COMPOSITION_CHANGE                 |           0.6490 |                   0.5590 |        0.6740 |        0.5176 |        0.2116 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | B2_RAW_COMPOSITIONS                   |           0.6252 |                   0.5329 |        0.6063 |        0.4441 |        0.2313 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | B3_MOLECULAR_FLUXES                   |           0.6347 |                   0.5610 |        0.6647 |        0.4533 |        0.2262 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | B4_ADJACENT_H                         |           0.6987 |                   0.6023 |        0.6891 |        0.5510 |        0.2000 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | B5_PREFIX_ATTRACTOR_GEOMETRY          |           0.6809 |                   0.6213 |        0.7055 |        0.5371 |        0.2010 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | B6_TIME_ONLY                          |           0.6446 |                   0.5901 |        0.6839 |        0.5133 |        0.2103 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | MAJORITY_DUMMY                        |           0.6425 |                   0.5000 |        0.5000 |        0.3575 |        0.2304 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | P1_PHIRL_EMERGENCE_COMPLETED_FIT      |           0.6563 |                   0.5731 |        0.6819 |        0.5241 |        0.2060 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY |           0.6434 |                   0.5700 |        0.6778 |        0.5270 |        0.2117 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | B1_COMPOSITION_CHANGE                 |           0.6091 |                   0.5450 |        0.6046 |        0.4792 |        0.2314 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | B2_RAW_COMPOSITIONS                   |           0.5757 |                   0.5132 |        0.5316 |        0.4132 |        0.2448 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | B3_MOLECULAR_FLUXES                   |           0.6137 |                   0.5204 |        0.5846 |        0.4500 |        0.2318 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | B4_ADJACENT_H                         |           0.6667 |                   0.5709 |        0.6545 |        0.5290 |        0.2175 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | B5_PREFIX_ATTRACTOR_GEOMETRY          |           0.6836 |                   0.6120 |        0.6855 |        0.5426 |        0.2133 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | B6_TIME_ONLY                          |           0.6198 |                   0.5432 |        0.6032 |        0.5021 |        0.2297 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | MAJORITY_DUMMY                        |           0.6020 |                   0.5000 |        0.5000 |        0.3980 |        0.2397 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | P1_PHIRL_EMERGENCE_COMPLETED_FIT      |           0.6361 |                   0.5421 |        0.6290 |        0.5195 |        0.2269 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY |           0.6143 |                   0.5346 |        0.6037 |        0.5116 |        0.2287 |

### Registered comparisons

| targetId                                   | candidateId   | comparisonFamily   | referenceModelId                      | comparatorModelId                      |   referenceMedianAccuracy |   comparatorMedianAccuracy |   mannWhitneyP |   holmAdjustedMannWhitneyP |   matrixBootstrapObservedDifference |   matrixBootstrapLower95 |   matrixBootstrapUpper95 |
|:-------------------------------------------|:--------------|:-------------------|:--------------------------------------|:---------------------------------------|--------------------------:|---------------------------:|---------------:|---------------------------:|------------------------------------:|-------------------------:|-------------------------:|
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PAPER              | P1_PHIRL_EMERGENCE_COMPLETED_FIT      | B1_COMPOSITION_CHANGE                  |                   0.65634 |                    0.64900 |        0.67758 |                    1.00000 |                             0.00721 |                  0.00073 |                  0.01418 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PAPER              | P1_PHIRL_EMERGENCE_COMPLETED_FIT      | B2_RAW_COMPOSITIONS                    |                   0.65634 |                    0.62520 |        0.08897 |                    1.00000 |                             0.03555 |                  0.01131 |                  0.05998 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PAPER              | P1_PHIRL_EMERGENCE_COMPLETED_FIT      | B3_MOLECULAR_FLUXES                    |                   0.65634 |                    0.63471 |        0.30749 |                    1.00000 |                             0.01751 |                  0.00325 |                  0.03144 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PAPER              | P1_PHIRL_EMERGENCE_COMPLETED_FIT      | MAJORITY_DUMMY                         |                   0.65634 |                    0.64251 |        0.47268 |                    1.00000 |                             0.03602 |                 -0.00416 |                  0.07918 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | MAJORITY_DUMMY                         |                   0.64343 |                    0.64251 |        0.96985 |                    1.00000 |                             0.02702 |                 -0.01506 |                  0.07261 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B6_TIME_ONLY                           |                   0.64343 |                    0.64459 |        0.79134 |                    1.00000 |                            -0.00276 |                 -0.01762 |                  0.01242 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B1_COMPOSITION_CHANGE                  |                   0.64343 |                    0.64900 |        0.79134 |                    1.00000 |                            -0.00179 |                 -0.01724 |                  0.01229 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B2_RAW_COMPOSITIONS                    |                   0.64343 |                    0.62520 |        0.21229 |                    1.00000 |                             0.02655 |                 -0.00045 |                  0.05462 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B3_MOLECULAR_FLUXES                    |                   0.64343 |                    0.63471 |        0.67758 |                    1.00000 |                             0.00851 |                 -0.00856 |                  0.02474 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B4_ADJACENT_H                          |                   0.64343 |                    0.69867 |        0.12122 |                    1.00000 |                            -0.03634 |                 -0.05316 |                 -0.02088 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B5_PREFIX_ATTRACTOR_GEOMETRY           |                   0.64343 |                    0.68094 |        0.07566 |                    1.00000 |                            -0.05826 |                 -0.08772 |                 -0.03059 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | INCREMENTAL        | P2_PLUS_B4                            | B4_ADJACENT_H                          |                   0.67645 |                    0.69867 |        0.85011 |                    1.00000 |                            -0.01140 |                 -0.02263 |                 -0.00125 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | INCREMENTAL        | P2_PLUS_B5                            | B5_PREFIX_ATTRACTOR_GEOMETRY           |                   0.69827 |                    0.68094 |        0.67758 |                    1.00000 |                             0.00485 |                 -0.00173 |                  0.01236 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | NEGATIVE_CONTROL   | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | NC1_WITHIN_PREFIX_TEMPORAL_PERMUTATION |                   0.64343 |                    0.64274 |        0.67758 |                    1.00000 |                            -0.00618 |                 -0.01947 |                  0.00587 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | NEGATIVE_CONTROL   | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | NC2_MATRIX_LABEL_PERMUTATION           |                   0.64343 |                    0.55012 |        0.01402 |                    0.22431 |                             0.07525 |                  0.03555 |                  0.11486 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_2   | NEGATIVE_CONTROL   | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B7_RANDOM_MATCHED_SHAPE                |                   0.64343 |                    0.65512 |        0.73373 |                    1.00000 |                            -0.00290 |                 -0.01826 |                  0.01103 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PAPER              | P1_PHIRL_EMERGENCE_COMPLETED_FIT      | B1_COMPOSITION_CHANGE                  |                   0.63609 |                    0.60905 |        0.24132 |                    1.00000 |                             0.01987 |                 -0.00108 |                  0.04377 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PAPER              | P1_PHIRL_EMERGENCE_COMPLETED_FIT      | B2_RAW_COMPOSITIONS                    |                   0.63609 |                    0.57571 |        0.00728 |                    0.10927 |                             0.06060 |                  0.03886 |                  0.08209 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PAPER              | P1_PHIRL_EMERGENCE_COMPLETED_FIT      | B3_MOLECULAR_FLUXES                    |                   0.63609 |                    0.61374 |        0.08897 |                    1.00000 |                             0.02542 |                  0.01309 |                  0.03841 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PAPER              | P1_PHIRL_EMERGENCE_COMPLETED_FIT      | MAJORITY_DUMMY                         |                   0.63609 |                    0.60203 |        0.30749 |                    1.00000 |                             0.02607 |                 -0.00740 |                  0.06344 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | MAJORITY_DUMMY                         |                   0.61434 |                    0.60203 |        0.62318 |                    1.00000 |                             0.00959 |                 -0.02898 |                  0.04938 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B6_TIME_ONLY                           |                   0.61434 |                    0.61981 |        0.73373 |                    1.00000 |                            -0.00454 |                 -0.02347 |                  0.01314 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B1_COMPOSITION_CHANGE                  |                   0.61434 |                    0.60905 |        0.79134 |                    1.00000 |                             0.00338 |                 -0.01602 |                  0.02292 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B2_RAW_COMPOSITIONS                    |                   0.61434 |                    0.57571 |        0.02575 |                    0.36047 |                             0.04411 |                  0.01644 |                  0.06910 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B3_MOLECULAR_FLUXES                    |                   0.61434 |                    0.61374 |        0.30749 |                    1.00000 |                             0.00893 |                 -0.01478 |                  0.02883 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B4_ADJACENT_H                          |                   0.61434 |                    0.66669 |        0.12122 |                    1.00000 |                            -0.04591 |                 -0.07564 |                 -0.01987 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | PROSPECTIVE        | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B5_PREFIX_ATTRACTOR_GEOMETRY           |                   0.61434 |                    0.68363 |        0.00580 |                    0.09273 |                            -0.06360 |                 -0.09934 |                 -0.03097 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | INCREMENTAL        | P2_PLUS_B4                            | B4_ADJACENT_H                          |                   0.64165 |                    0.66669 |        0.52052 |                    1.00000 |                            -0.02113 |                 -0.03767 |                 -0.00638 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | INCREMENTAL        | P2_PLUS_B5                            | B5_PREFIX_ATTRACTOR_GEOMETRY           |                   0.66648 |                    0.68363 |        0.21229 |                    1.00000 |                            -0.01793 |                 -0.03279 |                 -0.00408 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | NEGATIVE_CONTROL   | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | NC1_WITHIN_PREFIX_TEMPORAL_PERMUTATION |                   0.61434 |                    0.63367 |        0.62318 |                    1.00000 |                            -0.01078 |                 -0.02878 |                  0.00265 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | NEGATIVE_CONTROL   | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | NC2_MATRIX_LABEL_PERMUTATION           |                   0.61434 |                    0.57996 |        0.03121 |                    0.40572 |                             0.03995 |                  0.00399 |                  0.07572 |
| F5_U2_PAPER_EUCLIDEAN_RECURRING_UNION_H090 | CANDIDATE_3   | NEGATIVE_CONTROL   | P2_PHIRL_EMERGENCE_FIRST_QUARTER_ONLY | B7_RANDOM_MATCHED_SHAPE                |                   0.61434 |                    0.61374 |        1.00000 |                    1.00000 |                            -0.00613 |                 -0.02630 |                  0.00989 |

### Leakage and controls

| candidateId   |   matrixIndex |   p1MaximumAbsoluteChange |   p1MeanAbsoluteChange | p1ResultExact   |   p1SharedCount | p2MaskExact   | p2ResultExact   | p2ValuesExact   | passed   | prefixClrExact   | prefixStatesExact   |
|:--------------|--------------:|--------------------------:|-----------------------:|:----------------|----------------:|:--------------|:----------------|:----------------|:---------|:-----------------|:--------------------|
| CANDIDATE_2   |             0 |                   5.06016 |               0.766762 | False           |             205 | True          | True            | True            | True     | True             | True                |
| CANDIDATE_2   |            24 |                   9.28827 |               0.829021 | False           |             290 | True          | True            | True            | True     | True             | True                |
| CANDIDATE_2   |            49 |                   8.57498 |               0.842181 | False           |             274 | True          | True            | True            | True     | True             | True                |
| CANDIDATE_2   |            74 |                   3.73953 |               0.592862 | False           |             228 | True          | True            | True            | True     | True             | True                |
| CANDIDATE_3   |             0 |                   4.95854 |               0.777111 | False           |             229 | True          | True            | True            | True     | True             | True                |
| CANDIDATE_3   |            24 |                   2.74378 |               0.684624 | False           |             266 | True          | True            | True            | True     | True             | True                |
| CANDIDATE_3   |            49 |                  26.9561  |               0.963906 | False           |             269 | True          | True            | True            | True     | True             | True                |
| CANDIDATE_3   |            74 |                   2.64186 |               0.579773 | False           |             262 | True          | True            | True            | True     | True             | True                |

The completed-fit feature is explicitly future-dependent. Every prefix-only suffix audit had to pass before a prospective gate was evaluated. The oracle was excluded from ordinary comparisons and every promotion decision.

## Technical-assurance amendments

Two narrowly bounded repairs were required and remain visible in `technical_amendment_ledger.csv` and `failure_ledger.csv`. `L13-TA-001` corrected target-specific oracle cache routing before any MLP fit or prediction outcome; the fresh-cache rerun reproduced every pre-model feature artifact byte-for-byte. `L13-TA-002` replaced a removed Matplotlib boxplot keyword after all scientific outcomes were complete. Its fresh-cache rerun was required to reproduce every registered scientific table, feature cache, prediction, metric, gate, and classification exactly before any figure or report was released. The failed attempts remain quarantined under `/cache`; no target, feature, split, model, prediction, metric, test, gate, or classification changed.

## Illustrated results

1. ![Figure 5 baseline arithmetic clue](figures/figure_01_baseline_arithmetic_clue.png)
2. ![Target availability](figures/figure_02_target_availability.png)
3. ![Whole and suffix prevalence](figures/figure_03_whole_suffix_prevalence.png)
4. ![First-onset availability](figures/figure_04_first_onset_availability.png)
5. ![Representative target sequences](figures/figure_05_representative_target_sequences.png)
6. ![Completed-fit and prefix-only PhiRL](figures/figure_06_completed_vs_prefix_phirl.png)
7. ![Reconstructed Figure 5 accuracy](figures/figure_07_paper_accuracy_boxplots.png)
8. ![Robust metrics](figures/figure_08_robust_metrics.png)
9. ![Incremental-value comparisons](figures/figure_09_incremental_value.png)
10. ![Future dependence and invariance](figures/figure_10_future_dependence.png)
11. ![Negative controls](figures/figure_11_negative_controls.png)
12. ![Candidate agreement](figures/figure_12_candidate_agreement.png)
13. ![Decision matrix](figures/figure_13_decision_matrix.png)
14. ![Promotion decision tree](figures/figure_14_promotion_decision_tree.png)

## Validation

PASS: 400/400 target replays, 2400/2400 feature replays, 28 registered actual-model replay rows, and 2000 immutable prior files with 0 mismatches; 2 value-preserving technical amendments recorded.

The repository lock was clean and matched `origin/eidosoma/groups/42`; all 16 mandatory fixtures passed. The immutable baseline excluded only append-only S19 root ledgers and L13's own new directory. Exact U2 replay required identical trajectory states, labels, scores, and scoring centroids. Actual-model replay used identical initial weights, histories, and predictions for every registered model at repetition zero. Derived comparison tables were regenerated exactly.

`FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW.md` is retained as V1 for the 14 generated L13 figures. The human-requested V2, `FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md`, instead records my panel-by-panel reading of the input paper's Figures 1–6, caption meaning, visible values, operational implications, ambiguities, Table 1 conflicts, and a manual verification checklist against frozen paper/native-image hashes.

## Commands

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l13.py tests/e01/test_s16_prediction_reconstruction.py tests/e01/test_s19_l10.py tests/e01/test_s19_l11r.py
PYTHONPATH=src python scripts/e01/run_s19_l13.py prepare
PYTHONPATH=src python scripts/e01/run_s19_l13.py geometry
PYTHONPATH=src python scripts/e01/run_s19_l13.py benchmark
PYTHONPATH=src python scripts/e01/run_s19_l13.py execute
PYTHONPATH=src python scripts/e01/run_s19_l13.py finalize
```

## Runtime and dependencies

- CPU scientific seconds: `1383.279`; execution wall seconds: `1384.954`.
- Workers: at most 8; numeric-library threads: one per worker; GPU: not used.
- Python `3.13.14`, NumPy `2.4.6`, pandas `2.3.3`, SciPy `1.18.0`, scikit-learn `1.9.0`, PyTorch `2.11.0+cu128`.

## Caveats, blockers, and failed assumptions

- L13 is adaptive: the same matrices informed earlier label work, so even a promotable result requires untouched confirmation.
- Both targets are defined from completed trajectories. This makes the target task itself retrospective, even when P2 input is suffix-independent.
- L12's native-figure measurements are approximate and do not provide numeric box/whisker endpoints.
- Repeated splits overlap in matrix membership; they are paper-facing diagnostics, not ten independent experiments.
- No favorable candidate or target pooling is allowed; candidate disagreement is a failure.
- L10/L11R remain negative for the separate Table 1 88% fingerprint. L13 does not reinterpret those results.
- S18 prospective-prediction and causal-control non-support remains unchanged unless a later untouched confirmation is separately authorized and passes its own gates.

## Artifact and provenance index

The machine-readable target, feature, split, training, prediction, metric, comparison, control, leakage, gate, classification, runtime, storage, regeneration, and hash files are listed in `artifact_manifest.json`. Large disposable tensors stayed under `/cache/e01_s19_l13` and are not collectible artifacts. Repository code stayed in Git.

## Recommended next action

Return to mandatory human review. Do not begin S20, E02, a confirmation dataset, intervention analysis, report bundle, or another S19 loop automatically.
