# S19-L36 — Independent-Lineage Basin Transfer Audit

## Chief/human handoff

- **Step:** `E01-S19-L36-INDEPENDENT-LINEAGE-BASIN-TRANSFER-AUDIT-v1.0.0`
- **Status:** complete under the authorized L19–L42 sequence.
- **Classifications:** `TARGET_BASIN_LINEAGE_SPECIFIC`, `CURRENT_COMMITTOR_TARGET_NOT_NETWORK_STABLE`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** two independently seeded reference lineages per 280 frozen state/matrix/candidate units; exact beta and initial-state reuse; exact discrete/path original-target replay of all 53,760 H32/H8 branch streams with the recorded TA01 float64 numerical-equivalence contract; independent full lineage and rescore regeneration; 4,096 matrix bootstraps; immutable/runtime/storage/artifact hashes.
- **Recommended next action:** `MULTILINEAGE_ANY_ATTRACTOR_ENTRY_TARGET_CONSTRUCTION`.

## Question

Is the basin used by the empirical committor a reproducible property of the catalytic network, or a trajectory-specific object reconstructed from the same completed lineage being explained? L36 changes only target provenance. For every frozen state it generates two independent 100-fission reference lineages under the same beta, initial state and candidate semantics, applies the unchanged L23 target construction, and rescored the exact existing H32/H8 stochastic paths against ORIGINAL, REFERENCE_A and REFERENCE_B targets.

## Availability and basin agreement

| evaluationCohort   | candidateId       |   states |   referenceAEligibleFraction |   referenceBEligibleFraction |   referenceAAtRiskFraction |   referenceBAtRiskFraction |   referenceCentroidAgreementFraction |   referenceCentroidHMean |   referenceCentroidHMedian | availabilityGatePassed   | centroidAgreementGatePassed   |
|:-------------------|:------------------|---------:|-----------------------------:|-----------------------------:|---------------------------:|---------------------------:|-------------------------------------:|-------------------------:|---------------------------:|:-------------------------|:------------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 |       50 |                            1 |                            1 |                       0.96 |                      0.98  |                                 0.74 |                 0.831637 |                   0.963543 | True                     | False                         |
| L28_VALIDATION     | S12F-CANDIDATE-03 |       50 |                            1 |                            1 |                       0.94 |                      1     |                                 0.6  |                 0.70595  |                   0.94397  | True                     | False                         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |       40 |                            1 |                            1 |                       1    |                      0.975 |                                 0.75 |                 0.805128 |                   0.977911 | True                     | False                         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |       40 |                            1 |                            1 |                       1    |                      0.975 |                                 0.7  |                 0.805822 |                   0.969972 | True                     | False                         |

Reference-centroid agreement:

| evaluationCohort   | candidateId       |   states |    meanH |   medianH |   strictH090Agreement |
|:-------------------|:------------------|---------:|---------:|----------:|----------------------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 |       50 | 0.831637 |  0.963543 |                  0.74 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |       50 | 0.70595  |  0.94397  |                  0.6  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |       40 | 0.805128 |  0.977911 |                  0.75 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |       40 | 0.805822 |  0.969972 |                  0.7  |

Incomplete reference lineages and states already inside an independent target were retained as explicit ineligible units and were never replaced.

## Independent-target committor reliability

| evaluationCohort   | candidateId       | targetId    |   eligibleStates |    meanQ |   splitHalfSpearman |   intermediateStateCount |   correctedBetweenStateVariance |
|:-------------------|:------------------|:------------|-----------------:|---------:|--------------------:|-------------------------:|--------------------------------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 | ORIGINAL    |               50 | 0.251875 |            0.928578 |                       20 |                       0.10287   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | REFERENCE_A |               48 | 0.276855 |            0.925274 |                       19 |                       0.116331  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | REFERENCE_B |               49 | 0.249681 |            0.925113 |                       22 |                       0.0972905 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | ORIGINAL    |               50 | 0.227813 |            0.93977  |                       20 |                       0.081034  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | REFERENCE_A |               47 | 0.231882 |            0.958167 |                       23 |                       0.07011   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | REFERENCE_B |               50 | 0.195    |            0.94279  |                       21 |                       0.0600717 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | ORIGINAL    |               40 | 0.247461 |            0.930238 |                       26 |                       0.0716707 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | REFERENCE_A |               40 | 0.27832  |            0.958521 |                       27 |                       0.0738273 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | REFERENCE_B |               39 | 0.246795 |            0.924669 |                       25 |                       0.07109   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | ORIGINAL    |               40 | 0.317969 |            0.958141 |                       20 |                       0.120689  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | REFERENCE_A |               40 | 0.234961 |            0.94654  |                       20 |                       0.0900182 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | REFERENCE_B |               39 | 0.280449 |            0.934366 |                       20 |                       0.0969874 |

## Cross-lineage response and teacher transfer

| evaluationCohort   | candidateId       | comparisonId                       |   definedPairs |   spearman |   lower95 |   upper95 | rankGatePassed   |
|:-------------------|:------------------|:-----------------------------------|---------------:|-----------:|----------:|----------:|:-----------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | REFERENCE_A_VS_REFERENCE_B_H32     |             47 |   0.933005 | 0.859569  |  0.97274  | True             |
| L28_VALIDATION     | S12F-CANDIDATE-02 | REFERENCE_A_H8_VS_H32              |             48 |   0.803045 | 0.643818  |  0.896233 | True             |
| L28_VALIDATION     | S12F-CANDIDATE-02 | REFERENCE_B_H8_VS_H32              |             49 |   0.757782 | 0.582508  |  0.864457 | True             |
| L28_VALIDATION     | S12F-CANDIDATE-02 | ORIGINAL_H8_VS_REFERENCE_MEAN_H32  |             50 |   0.531591 | 0.256036  |  0.74118  | False            |
| L28_VALIDATION     | S12F-CANDIDATE-02 | ORIGINAL_H32_VS_REFERENCE_MEAN_H32 |             50 |   0.682417 | 0.446579  |  0.858381 | True             |
| L28_VALIDATION     | S12F-CANDIDATE-02 | REFERENCE_MEAN_H8_VS_H32           |             50 |   0.797935 | 0.646904  |  0.883216 | True             |
| L28_VALIDATION     | S12F-CANDIDATE-03 | REFERENCE_A_VS_REFERENCE_B_H32     |             47 |   0.395586 | 0.0577133 |  0.696444 | False            |
| L28_VALIDATION     | S12F-CANDIDATE-03 | REFERENCE_A_H8_VS_H32              |             47 |   0.735625 | 0.567746  |  0.84593  | True             |
| L28_VALIDATION     | S12F-CANDIDATE-03 | REFERENCE_B_H8_VS_H32              |             50 |   0.705055 | 0.53732   |  0.819923 | True             |
| L28_VALIDATION     | S12F-CANDIDATE-03 | ORIGINAL_H8_VS_REFERENCE_MEAN_H32  |             50 |   0.532016 | 0.260815  |  0.733846 | False            |
| L28_VALIDATION     | S12F-CANDIDATE-03 | ORIGINAL_H32_VS_REFERENCE_MEAN_H32 |             50 |   0.625683 | 0.358858  |  0.838183 | True             |
| L28_VALIDATION     | S12F-CANDIDATE-03 | REFERENCE_MEAN_H8_VS_H32           |             50 |   0.791824 | 0.649618  |  0.86969  | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | REFERENCE_A_VS_REFERENCE_B_H32     |             39 |   0.653848 | 0.361863  |  0.873314 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | REFERENCE_A_H8_VS_H32              |             40 |   0.72918  | 0.499997  |  0.891649 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | REFERENCE_B_H8_VS_H32              |             39 |   0.741283 | 0.518148  |  0.880869 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | ORIGINAL_H8_VS_REFERENCE_MEAN_H32  |             40 |   0.55878  | 0.271254  |  0.771965 | False            |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | ORIGINAL_H32_VS_REFERENCE_MEAN_H32 |             40 |   0.802529 | 0.589197  |  0.936997 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | REFERENCE_MEAN_H8_VS_H32           |             40 |   0.705935 | 0.456568  |  0.884211 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | REFERENCE_A_VS_REFERENCE_B_H32     |             39 |   0.585131 | 0.232769  |  0.884442 | False            |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | REFERENCE_A_H8_VS_H32              |             40 |   0.787665 | 0.620122  |  0.887937 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | REFERENCE_B_H8_VS_H32              |             39 |   0.864419 | 0.735012  |  0.932005 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | ORIGINAL_H8_VS_REFERENCE_MEAN_H32  |             40 |   0.692394 | 0.450837  |  0.842749 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | ORIGINAL_H32_VS_REFERENCE_MEAN_H32 |             40 |   0.771022 | 0.538138  |  0.943197 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | REFERENCE_MEAN_H8_VS_H32           |             40 |   0.850374 | 0.723349  |  0.917804 | True             |

## Frozen decision gates

| evaluationCohort   | candidateId       | availabilityPassed   | centroidAgreementPointPassed   | centroidAgreementLowerPassed   | referenceH32ReliabilityPassed   | referenceAReferenceBH32RankPassed   | correspondingReferenceH8Passed   | originalH8TransferPassed   | independentBasinStable   | originalShootingTransfers   |
|:-------------------|:------------------|:---------------------|:-------------------------------|:-------------------------------|:--------------------------------|:------------------------------------|:---------------------------------|:---------------------------|:-------------------------|:----------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | True                 | False                          | False                          | True                            | True                                | True                             | False                      | False                    | False                       |
| L28_VALIDATION     | S12F-CANDIDATE-03 | True                 | False                          | False                          | True                            | False                               | True                             | False                      | False                    | False                       |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | True                 | False                          | False                          | True                            | True                                | True                             | False                      | False                    | False                       |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | True                 | False                          | False                          | True                            | False                               | True                             | True                       | False                    | False                       |

The classifications are TARGET_BASIN_LINEAGE_SPECIFIC, CURRENT_COMMITTOR_TARGET_NOT_NETWORK_STABLE, NOT_PROMOTABLE_AS_CONFIRMED. A stable independent target would still be completed-lineage-conditioned; a transferring H8 coordinate would still use forward stochastic shooting. Neither result is a past-observable early-warning marker.

## Validation and provenance

- Repository lock: `f3816e1aabe530d014e13e29fdbcf6584cef0f06`.
- Failed attempt 01 remains recorded. TA01 changed only the score-extrema replay comparison from bit equality to finite-mask equality plus absolute and relative error <= `1e-12` and ULP distance <= `16`; labels, entry times, clocks, branch paths and all later scientific calculations were unchanged.
- Workers: `8` with one numerical-library thread per worker; GPU hours `0`.
- Wall time: `564.74` seconds.
- Reference trajectories generated: `560` plus the same full regeneration scope.
- Unique frozen H32/H8 branch streams rescored: `53760`; new branch streams: `0`.
- No matrix, initial state, target threshold, target construction, branch horizon, simulator or incomplete unit was changed or replaced.

## Limitations

REFERENCE_A and REFERENCE_B are full-trajectory retrospective constructions. Two references can reveal lineage dependence but cannot exhaust a genuinely multimodal attractor landscape. The same original state is evaluated against target basins that may not be reachable from it, which is why current-inside and availability status are explicit. One state per matrix still prevents within-matrix ordering. This audit does not test PhiRL, the paper's prediction claim, intervention, or causal control.

## Next boundary

L36 is frozen. The standing authorization permits `MULTILINEAGE_ANY_ATTRACTOR_ENTRY_TARGET_CONSTRUCTION` as the only next loop. S20, E02, author contact, interventions, reactive-current work and report generation remain inactive.
