# S19-L52 Full Results — Shooting-Residual Regime Compression

## Top summary

- **Research step:** `E01-S19-L52-SHOOTING-RESIDUAL-REGIME-COMPRESSION-v1.0.0`
- **Completion status:** complete; additive exploratory analysis-only evidence
- **Artifacts written:** exact L50/L51 branch-half replay, three locked regime-hazard models, A-to-B and B-to-A fits, transition and F4/F8/F12 event proper scores, q ranks, residual reliability, 4,096 matrix bootstraps, 512 whole-matrix permutations, six figures, report and hash manifests
- **Validation:** PASS — immutable S01–L51 baseline; ten fixtures; exact 800-state/51,200-branch/614,400-transition replay; target-state exclusion from every matrix-transfer fit; heldout-half exclusion; zero-overlap analysis seeds; two exact analysis/report passes; runtime, storage and artifact hashes
- **Outcome classification:** `MATRIX_LEVEL_REGIME_DYNAMICS_TRANSFER_ACROSS_STATES`, `CURRENT_STATE_SPECIFIC_REGIME_DYNAMICS_REQUIRED`, `SHOOTING_COMMITTOR_COMPRESSIBLE_TO_STATE_LOCAL_DURATION_HAZARDS`, `BRANCH_DERIVED_NOT_PAST_OBSERVABLE`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Lay summary:** L52 uses one independent half of the already simulated futures to learn a simple hereditary-regime transition law and asks whether that law predicts the other half. It separates a matrix-wide law learned from other states from a law learned at the exact current state.
- **Recommended next action:** `L53_PAST_OBSERVABLE_STATE_LOCAL_HAZARD_PROXY` under the bounded autonomous authorization through L65. No new branch begins automatically inside L52; S20, E02, author contact, Phi and interventions remain inactive.

## Frozen design

L52 changes no scientific target. Strict parent/daughter `H>0.9` remains inheritance; the primary F12 event remains the first future break followed by three consecutive inherited fissions. The exact L50 branch halves are crossfit in both directions. `MATRIX_OTHER_LANDMARK_SEMIMARKOV` excludes the target state and learns only from the same matrix's other four landmarks. `STATE_LOCAL_SEMIMARKOV` learns from the fitting half at the target state. Both use one fixed four-pseudotransition cell prior anchored to the exact L51 pooled duration table. The scoring half never enters the fit.

## Heldout-half transition scores

| matrixRole   | candidateId       | modelId                          |   matrices |   equalMatrixMeanLogLoss |   equalMatrixMeanBrier |   matrixSdLogLoss |
|:-------------|:------------------|:---------------------------------|-----------:|-------------------------:|-----------------------:|------------------:|
| VALIDATION   | S12F-CANDIDATE-02 | STATE_LOCAL_SEMIMARKOV           |         40 |                0.3008046 |              0.0887835 |         0.1806042 |
| VALIDATION   | S12F-CANDIDATE-02 | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |                0.3029873 |              0.0893644 |         0.1804313 |
| VALIDATION   | S12F-CANDIDATE-02 | L51_POOLED_SEMIMARKOV            |         40 |                0.3170394 |              0.0920276 |         0.1879899 |
| VALIDATION   | S12F-CANDIDATE-03 | STATE_LOCAL_SEMIMARKOV           |         40 |                0.3067901 |              0.0913206 |         0.1866134 |
| VALIDATION   | S12F-CANDIDATE-03 | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |                0.3079457 |              0.0916664 |         0.1856990 |
| VALIDATION   | S12F-CANDIDATE-03 | L51_POOLED_SEMIMARKOV            |         40 |                0.3226399 |              0.0943851 |         0.1888861 |

## Heldout-half F12 joint-event scores

| matrixRole   | candidateId       |   horizon | targetType       | modelId                          |   matrices |   statesTimesDirections |   equalMatrixMeanBranchLogLoss |   equalMatrixMeanBranchBrier |     qRmse |   qSpearmanPooledDirections |
|:-------------|:------------------|----------:|:-----------------|:---------------------------------|-----------:|------------------------:|-------------------------------:|-----------------------------:|----------:|----------------------------:|
| VALIDATION   | S12F-CANDIDATE-02 |        12 | JOINT_BREAK_RUN3 | L51_POOLED_SEMIMARKOV            |         40 |                     400 |                      0.6330212 |                    0.2209069 | 0.2046096 |                   0.5797907 |
| VALIDATION   | S12F-CANDIDATE-02 |        12 | JOINT_BREAK_RUN3 | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |                     400 |                      0.5762578 |                    0.2016307 | 0.1538642 |                   0.7686576 |
| VALIDATION   | S12F-CANDIDATE-02 |        12 | JOINT_BREAK_RUN3 | STATE_LOCAL_SEMIMARKOV           |         40 |                     400 |                      0.5480583 |                    0.1892569 | 0.1092980 |                   0.8925386 |
| VALIDATION   | S12F-CANDIDATE-03 |        12 | JOINT_BREAK_RUN3 | L51_POOLED_SEMIMARKOV            |         40 |                     400 |                      0.6306394 |                    0.2199103 | 0.2095099 |                   0.5801657 |
| VALIDATION   | S12F-CANDIDATE-03 |        12 | JOINT_BREAK_RUN3 | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |                     400 |                      0.5614027 |                    0.1951941 | 0.1434584 |                   0.8181354 |
| VALIDATION   | S12F-CANDIDATE-03 |        12 | JOINT_BREAK_RUN3 | STATE_LOCAL_SEMIMARKOV           |         40 |                     400 |                      0.5356878 |                    0.1835767 | 0.0976706 |                   0.9244958 |

## Registered proper-score comparisons

| metricType      | candidateId       | comparisonId                       | modelId                          | referenceModelId                 |   matrices |   logLossImprovement |   logLossImprovementLower95 |   logLossImprovementUpper95 |   fractionBootstrapPositive |
|:----------------|:------------------|:-----------------------------------|:---------------------------------|:---------------------------------|-----------:|---------------------:|----------------------------:|----------------------------:|----------------------------:|
| TRANSITION      | S12F-CANDIDATE-02 | MATRIX_TRANSFER_BEYOND_POOLED      | MATRIX_OTHER_LANDMARK_SEMIMARKOV | L51_POOLED_SEMIMARKOV            |         40 |            0.0140521 |                   0.0072820 |                   0.0222637 |                   1.0000000 |
| TRANSITION      | S12F-CANDIDATE-02 | STATE_LOCAL_BEYOND_MATRIX_TRANSFER | STATE_LOCAL_SEMIMARKOV           | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |            0.0021827 |                  -0.0001841 |                   0.0049859 |                   0.9626465 |
| TRANSITION      | S12F-CANDIDATE-02 | STATE_LOCAL_BEYOND_POOLED          | STATE_LOCAL_SEMIMARKOV           | L51_POOLED_SEMIMARKOV            |         40 |            0.0162348 |                   0.0099646 |                   0.0239384 |                   1.0000000 |
| TRANSITION      | S12F-CANDIDATE-03 | MATRIX_TRANSFER_BEYOND_POOLED      | MATRIX_OTHER_LANDMARK_SEMIMARKOV | L51_POOLED_SEMIMARKOV            |         40 |            0.0146942 |                   0.0089113 |                   0.0207371 |                   1.0000000 |
| TRANSITION      | S12F-CANDIDATE-03 | STATE_LOCAL_BEYOND_MATRIX_TRANSFER | STATE_LOCAL_SEMIMARKOV           | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |            0.0011556 |                  -0.0011653 |                   0.0039813 |                   0.8095703 |
| TRANSITION      | S12F-CANDIDATE-03 | STATE_LOCAL_BEYOND_POOLED          | STATE_LOCAL_SEMIMARKOV           | L51_POOLED_SEMIMARKOV            |         40 |            0.0158498 |                   0.0105843 |                   0.0214736 |                   1.0000000 |
| F12_JOINT_EVENT | S12F-CANDIDATE-02 | MATRIX_TRANSFER_BEYOND_POOLED      | MATRIX_OTHER_LANDMARK_SEMIMARKOV | L51_POOLED_SEMIMARKOV            |         40 |            0.0567634 |                   0.0325460 |                   0.0853008 |                   1.0000000 |
| F12_JOINT_EVENT | S12F-CANDIDATE-02 | STATE_LOCAL_BEYOND_MATRIX_TRANSFER | STATE_LOCAL_SEMIMARKOV           | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |            0.0281995 |                   0.0170287 |                   0.0400423 |                   1.0000000 |
| F12_JOINT_EVENT | S12F-CANDIDATE-02 | STATE_LOCAL_BEYOND_POOLED          | STATE_LOCAL_SEMIMARKOV           | L51_POOLED_SEMIMARKOV            |         40 |            0.0849629 |                   0.0602792 |                   0.1119651 |                   1.0000000 |
| F12_JOINT_EVENT | S12F-CANDIDATE-03 | MATRIX_TRANSFER_BEYOND_POOLED      | MATRIX_OTHER_LANDMARK_SEMIMARKOV | L51_POOLED_SEMIMARKOV            |         40 |            0.0692368 |                   0.0427457 |                   0.0983183 |                   1.0000000 |
| F12_JOINT_EVENT | S12F-CANDIDATE-03 | STATE_LOCAL_BEYOND_MATRIX_TRANSFER | STATE_LOCAL_SEMIMARKOV           | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |            0.0257149 |                   0.0158783 |                   0.0367195 |                   1.0000000 |
| F12_JOINT_EVENT | S12F-CANDIDATE-03 | STATE_LOCAL_BEYOND_POOLED          | STATE_LOCAL_SEMIMARKOV           | L51_POOLED_SEMIMARKOV            |         40 |            0.0949517 |                   0.0692872 |                   0.1232660 |                   1.0000000 |

## Independent-half q ranking

| candidateId       | direction   | modelId                          |   matrices |   states |   qSpearman |   qSpearmanLower95 |   qSpearmanUpper95 |   centeredQSpearman |   centeredQSpearmanLower95 |   centeredQSpearmanUpper95 |
|:------------------|:------------|:---------------------------------|-----------:|---------:|------------:|-------------------:|-------------------:|--------------------:|---------------------------:|---------------------------:|
| S12F-CANDIDATE-02 | A_TO_B      | L51_POOLED_SEMIMARKOV            |         40 |      200 |   0.5846090 |          0.4544474 |          0.6895600 |           0.3451223 |                  0.1790458 |                  0.5051268 |
| S12F-CANDIDATE-02 | A_TO_B      | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |      200 |   0.7694965 |          0.6428623 |          0.8518326 |           0.0815856 |                 -0.1482299 |                  0.3014680 |
| S12F-CANDIDATE-02 | A_TO_B      | STATE_LOCAL_SEMIMARKOV           |         40 |      200 |   0.8904866 |          0.8303615 |          0.9278644 |           0.6959307 |                  0.5905533 |                  0.7717581 |
| S12F-CANDIDATE-02 | B_TO_A      | L51_POOLED_SEMIMARKOV            |         40 |      200 |   0.5721110 |          0.4431260 |          0.6761364 |           0.2947974 |                  0.1216871 |                  0.4469846 |
| S12F-CANDIDATE-02 | B_TO_A      | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |      200 |   0.7672767 |          0.6432734 |          0.8448913 |           0.0512978 |                 -0.1592911 |                  0.2544880 |
| S12F-CANDIDATE-02 | B_TO_A      | STATE_LOCAL_SEMIMARKOV           |         40 |      200 |   0.8941018 |          0.8403235 |          0.9287171 |           0.6772663 |                  0.5575449 |                  0.7697630 |
| S12F-CANDIDATE-03 | A_TO_B      | L51_POOLED_SEMIMARKOV            |         40 |      200 |   0.5638238 |          0.4104408 |          0.6823349 |           0.2897946 |                  0.1196385 |                  0.4459205 |
| S12F-CANDIDATE-03 | A_TO_B      | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |      200 |   0.8214935 |          0.7072412 |          0.8863304 |           0.2274075 |                  0.0137406 |                  0.4249012 |
| S12F-CANDIDATE-03 | A_TO_B      | STATE_LOCAL_SEMIMARKOV           |         40 |      200 |   0.9242333 |          0.8760352 |          0.9514532 |           0.7558105 |                  0.6412875 |                  0.8381274 |
| S12F-CANDIDATE-03 | B_TO_A      | L51_POOLED_SEMIMARKOV            |         40 |      200 |   0.5962840 |          0.4511887 |          0.7054553 |           0.3112374 |                  0.1238678 |                  0.4894272 |
| S12F-CANDIDATE-03 | B_TO_A      | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |      200 |   0.8144133 |          0.6988546 |          0.8776301 |           0.1670150 |                 -0.0403825 |                  0.3580832 |
| S12F-CANDIDATE-03 | B_TO_A      | STATE_LOCAL_SEMIMARKOV           |         40 |      200 |   0.9270198 |          0.8845896 |          0.9497041 |           0.7630410 |                  0.6490640 |                  0.8381488 |

## Residual reliability

| candidateId       | modelId                          |   matrices |   states |   residualSplitHalfSpearman |   residualSplitHalfSpearmanLower95 |   residualSplitHalfSpearmanUpper95 |   centeredResidualSpearman |   centeredResidualSpearmanLower95 |   centeredResidualSpearmanUpper95 |
|:------------------|:---------------------------------|-----------:|---------:|----------------------------:|-----------------------------------:|-----------------------------------:|---------------------------:|----------------------------------:|----------------------------------:|
| S12F-CANDIDATE-02 | L51_POOLED_SEMIMARKOV            |         40 |      200 |                   0.8195774 |                          0.7458619 |                          0.8698686 |                  0.7086175 |                         0.6029220 |                         0.7896881 |
| S12F-CANDIDATE-02 | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |      200 |                   0.5038278 |                          0.3861645 |                          0.6129391 |                  0.6717297 |                         0.5616591 |                         0.7581678 |
| S12F-CANDIDATE-02 | STATE_LOCAL_SEMIMARKOV           |         40 |      200 |                  -0.5051014 |                         -0.6343733 |                         -0.3578200 |                 -0.5311012 |                        -0.6592031 |                        -0.3868561 |
| S12F-CANDIDATE-03 | L51_POOLED_SEMIMARKOV            |         40 |      200 |                   0.8881128 |                          0.8293171 |                          0.9253104 |                  0.8239286 |                         0.7436054 |                         0.8800438 |
| S12F-CANDIDATE-03 | MATRIX_OTHER_LANDMARK_SEMIMARKOV |         40 |      200 |                   0.6790618 |                          0.5686569 |                          0.7773045 |                  0.7714301 |                         0.6725233 |                         0.8470184 |
| S12F-CANDIDATE-03 | STATE_LOCAL_SEMIMARKOV           |         40 |      200 |                  -0.3187365 |                         -0.4803146 |                         -0.1420972 |                 -0.2755802 |                        -0.4499749 |                        -0.0871596 |

A positive centered residual correlation means that the same states are systematically under- or overpredicted in both independent halves, so the duration table has not compressed all reproducible state information.

## Scientific gates

| gateId                                | candidateId       | gateFamily      |   properScoreLower95 |   minimumQSpearman |   minimumQSpearmanLower95 |   residualUpper95 |   maximumPermutationP | passed   |
|:--------------------------------------|:------------------|:----------------|---------------------:|-------------------:|--------------------------:|------------------:|----------------------:|:---------|
| MATRIX_TRANSFER::S12F-CANDIDATE-02    | S12F-CANDIDATE-02 | MATRIX_TRANSFER |            0.0325460 |          0.7672767 |                 0.6428623 |       nan         |             0.0019493 | True     |
| STATE_SPECIFIC::S12F-CANDIDATE-02     | S12F-CANDIDATE-02 | STATE_SPECIFIC  |            0.0170287 |          0.8904866 |                 0.8303615 |        -0.3868561 |             0.0019493 | True     |
| COMPRESSION::S12F-CANDIDATE-02        | S12F-CANDIDATE-02 | COMPRESSION     |            0.0602792 |          0.8904866 |                 0.8303615 |        -0.3868561 |             0.0019493 | True     |
| MATRIX_TRANSFER::S12F-CANDIDATE-03    | S12F-CANDIDATE-03 | MATRIX_TRANSFER |            0.0427457 |          0.8144133 |                 0.6988546 |       nan         |             0.0019493 | True     |
| STATE_SPECIFIC::S12F-CANDIDATE-03     | S12F-CANDIDATE-03 | STATE_SPECIFIC  |            0.0158783 |          0.9242333 |                 0.8760352 |        -0.0871596 |             0.0019493 | True     |
| COMPRESSION::S12F-CANDIDATE-03        | S12F-CANDIDATE-03 | COMPRESSION     |            0.0692872 |          0.9242333 |                 0.8760352 |        -0.0871596 |             0.0019493 | True     |
| COMPLETE_CROSS_CANDIDATE_ADJUDICATION | BOTH              | COMPLETE        |          nan         |        nan         |               nan         |       nan         |           nan         | True     |

## Interpretation boundary

Matrix- or state-local branch-derived hazards are forward-shooting measurements. Even perfect compression would not make them past-observable biomarkers. A failure of compression does not eliminate a real committor; it means that binary duration hazards lose path ordering, evolving physical state or other future-ensemble information. This result cannot establish paper replication, a privileged attractor, functional memory, PhiID foresight, intervention efficacy or real chemistry.

## Runtime and provenance

- Repository lock: `dd12b547dc10baa415cff969cd890224adb63b4e`.
- Workers: `1` with one numerical-library thread; GPU hours: 0.
- Wall time: `83.240` minutes; CPU upper estimate: `1.387334` hours.
- New matrices, trajectories and branch streams: 0, 0 and 0.
- Frozen branch sequences/transitions: `51,200` / `614,400`.
- Crossfit directions: 2; matrix bootstraps: 4096; whole-matrix permutations: 512.

## Limitations

This adaptive loop reuses the L50 branch ensemble. The binary transition process compresses continuous composition and catalytic dynamics. Matrix-transfer models use simulated futures at other states and state-local models use simulated futures at the target; neither is operational without shooting. The four-pseudotransition prior and duration cap were fixed, not searched. Matrix resampling preserves dependence among landmarks and branch folds.
