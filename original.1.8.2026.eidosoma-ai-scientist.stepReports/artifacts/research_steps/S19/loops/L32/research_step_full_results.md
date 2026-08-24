# S19-L32 — Committor-Ordered Past-Only Transition-Tube Coordinate

## Chief/human handoff

- **Step:** `E01-S19-L32-COMMITTOR-ORDERED-PAST-ONLY-TRANSITION-TUBE-v1.0.0`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** `PAST_ONLY_TRANSITION_TUBE_COMMITTOR_COORDINATE_NON_SUPPORT`, `CONFIRMED_COMMITTOR_NOT_RECOVERED_BY_FROZEN_OBSERVED_PREFIX_TUBES`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** exact L27 representations for all 280 L28/L31 states; development-only candidate models; unchanged L28 validation and L31 confirmation; suffix invariance; temporal reversal; 512 development and evaluation permutations; 4,096 matrix bootstraps; exact model/report regeneration and artifact hashes.
- **Next bounded theme:** SINGLE_STATE_MEMORY_AND_PHASE_COORDINATE_AUDIT

## Frozen question and method

Can one observed 32-state prefix recover the reliable H32 committor without H8 shooting branches or a completed-run centroid predictor? The full view contains 11 past-only level/current channels; exact-H/recurrence and ordinary composition/dynamics views are separate controls. Models use only L28 development q and are evaluated unchanged on L28 validation and the previously untouched L31 confirmation cohort.

## Metrics

| variant           | evaluationCohort   | candidateId       | modelId                  |   states |   spearmanQHat |   brierScorePerBranch |   binomialLogLossPerBranch |   calibrationIntercept |   calibrationSlope |
|:------------------|:-------------------|:------------------|:-------------------------|---------:|---------------:|----------------------:|---------------------------:|-----------------------:|-------------------:|
| ORIGINAL          | L28_DEVELOPMENT    | S12F-CANDIDATE-02 | EXACT_H_TRANSITION_TUBE  |       50 |    0.994976    |             0.102703  |                   0.328558 |             0.0204164  |          1.02918   |
| ORIGINAL          | L28_DEVELOPMENT    | S12F-CANDIDATE-02 | FULL_TRANSITION_TUBE     |       50 |    0.998631    |             0.102686  |                   0.327842 |             0.00615087 |          1.01033   |
| ORIGINAL          | L28_DEVELOPMENT    | S12F-CANDIDATE-02 | LANDMARK_PRIOR           |       50 |    0.1472      |             0.18397   |                   0.551892 |             0          |          1         |
| ORIGINAL          | L28_DEVELOPMENT    | S12F-CANDIDATE-02 | ORDINARY_TRANSITION_TUBE |       50 |    0.998246    |             0.102698  |                   0.328217 |           nan          |        nan         |
| ORIGINAL          | L28_DEVELOPMENT    | S12F-CANDIDATE-03 | EXACT_H_TRANSITION_TUBE  |       50 |    0.992749    |             0.073095  |                   0.247164 |             0.0552578  |          1.05131   |
| ORIGINAL          | L28_DEVELOPMENT    | S12F-CANDIDATE-03 | FULL_TRANSITION_TUBE     |       50 |    0.994873    |             0.073062  |                   0.246157 |             0.0142583  |          1.01376   |
| ORIGINAL          | L28_DEVELOPMENT    | S12F-CANDIDATE-03 | LANDMARK_PRIOR           |       50 |    0.423076    |             0.139929  |                   0.439175 |             0          |          1         |
| ORIGINAL          | L28_DEVELOPMENT    | S12F-CANDIDATE-03 | ORDINARY_TRANSITION_TUBE |       50 |    0.992701    |             0.0730737 |                   0.246549 |           nan          |        nan         |
| ORIGINAL          | L28_VALIDATION     | S12F-CANDIDATE-02 | EXACT_H_TRANSITION_TUBE  |       50 |    0.0832412   |             0.256209  |                   1.36535  |           nan          |        nan         |
| ORIGINAL          | L28_VALIDATION     | S12F-CANDIDATE-02 | FULL_TRANSITION_TUBE     |       50 |    0.158793    |             0.344138  |                   1.24675  |           nan          |        nan         |
| ORIGINAL          | L28_VALIDATION     | S12F-CANDIDATE-02 | LANDMARK_PRIOR           |       50 |    0.238344    |             0.188344  |                   0.563766 |           nan          |        nan         |
| ORIGINAL          | L28_VALIDATION     | S12F-CANDIDATE-02 | ORDINARY_TRANSITION_TUBE |       50 |   -0.0280675   |             0.302342  |                   1.42135  |           nan          |        nan         |
| ORIGINAL          | L28_VALIDATION     | S12F-CANDIDATE-03 | EXACT_H_TRANSITION_TUBE  |       50 |    0.153598    |             0.257076  |                   1.25286  |           nan          |        nan         |
| ORIGINAL          | L28_VALIDATION     | S12F-CANDIDATE-03 | FULL_TRANSITION_TUBE     |       50 |    0.0122629   |             0.249246  |                   0.974371 |            -1.1364     |          0.0517299 |
| ORIGINAL          | L28_VALIDATION     | S12F-CANDIDATE-03 | LANDMARK_PRIOR           |       50 |    0.26106     |             0.187737  |                   0.559352 |            -0.675771   |          0.358035  |
| ORIGINAL          | L28_VALIDATION     | S12F-CANDIDATE-03 | ORDINARY_TRANSITION_TUBE |       50 |    0.0537162   |             0.318832  |                   1.41001  |           nan          |        nan         |
| ORIGINAL          | L31_CONFIRMATION   | S12F-CANDIDATE-02 | EXACT_H_TRANSITION_TUBE  |       40 |   -0.267769    |             0.442447  |                   2.75629  |           nan          |        nan         |
| ORIGINAL          | L31_CONFIRMATION   | S12F-CANDIDATE-02 | FULL_TRANSITION_TUBE     |       40 |   -0.221638    |             0.332896  |                   1.38556  |           nan          |        nan         |
| ORIGINAL          | L31_CONFIRMATION   | S12F-CANDIDATE-02 | LANDMARK_PRIOR           |       40 |    0.251541    |             0.185194  |                   0.555385 |           nan          |        nan         |
| ORIGINAL          | L31_CONFIRMATION   | S12F-CANDIDATE-02 | ORDINARY_TRANSITION_TUBE |       40 |    0.0256495   |             0.267848  |                   1.29897  |           nan          |        nan         |
| ORIGINAL          | L31_CONFIRMATION   | S12F-CANDIDATE-03 | EXACT_H_TRANSITION_TUBE  |       40 |    0.0873076   |             0.329669  |                   1.26454  |            -0.626032   |          0.0796867 |
| ORIGINAL          | L31_CONFIRMATION   | S12F-CANDIDATE-03 | FULL_TRANSITION_TUBE     |       40 |    0.0859933   |             0.273599  |                   1.04339  |           nan          |        nan         |
| ORIGINAL          | L31_CONFIRMATION   | S12F-CANDIDATE-03 | LANDMARK_PRIOR           |       40 |    0.183141    |             0.224061  |                   0.661753 |             0.008311   |          0.504022  |
| ORIGINAL          | L31_CONFIRMATION   | S12F-CANDIDATE-03 | ORDINARY_TRANSITION_TUBE |       40 |    0           |             0.28741   |                   1.33148  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L28_DEVELOPMENT    | S12F-CANDIDATE-02 | EXACT_H_TRANSITION_TUBE  |       50 |    0.107759    |             0.395883  |                   1.61207  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L28_DEVELOPMENT    | S12F-CANDIDATE-02 | FULL_TRANSITION_TUBE     |       50 |    0.308995    |             0.190882  |                   0.801901 |            -0.86395    |          0.174043  |
| TEMPORAL_REVERSAL | L28_DEVELOPMENT    | S12F-CANDIDATE-02 | ORDINARY_TRANSITION_TUBE |       50 |    0.152959    |             0.196711  |                   1.02143  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L28_DEVELOPMENT    | S12F-CANDIDATE-03 | EXACT_H_TRANSITION_TUBE  |       50 |    0.124335    |             0.263708  |                   1.50686  |            -1.29079    |          0.0216712 |
| TEMPORAL_REVERSAL | L28_DEVELOPMENT    | S12F-CANDIDATE-03 | FULL_TRANSITION_TUBE     |       50 |    0.453755    |             0.171025  |                   0.586237 |            -0.633303   |          0.313164  |
| TEMPORAL_REVERSAL | L28_DEVELOPMENT    | S12F-CANDIDATE-03 | ORDINARY_TRANSITION_TUBE |       50 |    0.121005    |             0.305115  |                   1.11498  |            -1.27694    |          0.0344582 |
| TEMPORAL_REVERSAL | L28_VALIDATION     | S12F-CANDIDATE-02 | EXACT_H_TRANSITION_TUBE  |       50 |   -0.132648    |             0.354205  |                   1.65213  |            -1.24209    |         -0.0871885 |
| TEMPORAL_REVERSAL | L28_VALIDATION     | S12F-CANDIDATE-02 | FULL_TRANSITION_TUBE     |       50 |    0.159754    |             0.290705  |                   1.01913  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L28_VALIDATION     | S12F-CANDIDATE-02 | ORDINARY_TRANSITION_TUBE |       50 |    0.13433     |             0.320735  |                   1.2517   |            -1.04569    |          0.0236592 |
| TEMPORAL_REVERSAL | L28_VALIDATION     | S12F-CANDIDATE-03 | EXACT_H_TRANSITION_TUBE  |       50 |   -0.000192359 |             0.358884  |                   1.61841  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L28_VALIDATION     | S12F-CANDIDATE-03 | FULL_TRANSITION_TUBE     |       50 |    0.00490515  |             0.264154  |                   0.954457 |           nan          |        nan         |
| TEMPORAL_REVERSAL | L28_VALIDATION     | S12F-CANDIDATE-03 | ORDINARY_TRANSITION_TUBE |       50 |   -0.0219289   |             0.339806  |                   1.51309  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L31_CONFIRMATION   | S12F-CANDIDATE-02 | EXACT_H_TRANSITION_TUBE  |       40 |    0.23047     |             0.293792  |                   1.29467  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L31_CONFIRMATION   | S12F-CANDIDATE-02 | FULL_TRANSITION_TUBE     |       40 |    0.0825857   |             0.27321   |                   1.00002  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L31_CONFIRMATION   | S12F-CANDIDATE-02 | ORDINARY_TRANSITION_TUBE |       40 |   -0.0889746   |             0.328814  |                   1.31731  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L31_CONFIRMATION   | S12F-CANDIDATE-03 | EXACT_H_TRANSITION_TUBE  |       40 |    0.102516    |             0.272131  |                   1.11585  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L31_CONFIRMATION   | S12F-CANDIDATE-03 | FULL_TRANSITION_TUBE     |       40 |   -0.0471273   |             0.30076   |                   1.21095  |           nan          |        nan         |
| TEMPORAL_REVERSAL | L31_CONFIRMATION   | S12F-CANDIDATE-03 | ORDINARY_TRANSITION_TUBE |       40 |   -0.0658093   |             0.336364  |                   1.51968  |           nan          |        nan         |

## Solution gates

| evaluationCohort   | candidateId       |   states |   primarySpearman |   primarySpearmanLower95 |   brierImprovementLowerVsLANDMARK_PRIOR |   brierImprovementLowerVsEXACT_H_TRANSITION_TUBE |   brierImprovementLowerVsORDINARY_TRANSITION_TUBE |   developmentPermutationP |   evaluationPermutationP |   temporalReversalSpearman | rankPassed   | incrementalBrierPassed   | developmentPermutationPassed   | evaluationPermutationPassed   | temporalReversalPassed   | suffixPassed   | cohortCandidateGatePassed   |
|:-------------------|:------------------|---------:|------------------:|-------------------------:|----------------------------------------:|-------------------------------------------------:|--------------------------------------------------:|--------------------------:|-------------------------:|---------------------------:|:-------------|:-------------------------|:-------------------------------|:------------------------------|:-------------------------|:---------------|:----------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 |       50 |         0.158793  |                -0.116606 |                               -0.234604 |                                       -0.177653  |                                        -0.0929668 |                  0.417154 |                 0.421053 |                 0.159754   | False        | False                    | False                          | False                         | False                    | True           | False                       |
| L28_VALIDATION     | S12F-CANDIDATE-03 |       50 |         0.0122629 |                -0.274182 |                               -0.133919 |                                       -0.0711324 |                                         0.026299  |                  0.883041 |                 0.900585 |                 0.00490515 | False        | False                    | False                          | False                         | True                     | True           | False                       |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |       40 |        -0.221638  |                -0.510004 |                               -0.235306 |                                        0.0393345 |                                        -0.134901  |                  1        |                 1        |                 0.0825857  | False        | False                    | False                          | False                         | False                    | True           | False                       |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |       40 |         0.0859933 |                -0.232698 |                               -0.12149  |                                       -0.0262044 |                                        -0.041844  |                  0.678363 |                 0.699805 |                -0.0471273  | False        | False                    | False                          | False                         | True                     | True           | False                       |

## Interpretation boundary

Even a passing result is conditioned on a retrospectively constructed target basin. It supports an organization-before-entry coordinate within that reconstructed task, not the paper authors' exact label, causal control, or biological validation. Branch-derived H8 values never enter the predictor.

## Runtime

- Repository lock: `6e844a980a55d8f29452997485a1080beb5f853b`.
- CPU float64; no GPU; wall seconds `164.652`.

## Autonomous boundary

L32 is frozen. S20, E02, author contact, interventions, reactive-current claims and report-bundle work remain inactive.
