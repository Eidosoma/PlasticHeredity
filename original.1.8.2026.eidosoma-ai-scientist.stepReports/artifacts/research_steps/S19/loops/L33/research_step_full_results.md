# S19-L33 — Single-State Operator-Memory and Phase Committor Coordinate

## Chief/human handoff

- **Step:** `E01-S19-L33-SINGLE-STATE-OPERATOR-MEMORY-PHASE-COMMITTOR-v1.0.0`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** `BASIN_BLIND_OPERATOR_MEMORY_COMMITTOR_NON_SUPPORT`, `BRANCH_SIGNAL_NOT_DISTILLED_FROM_LOW_DIMENSIONAL_OPERATOR_HISTORY`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Artifacts:** compact feature/replay tables, fitted models, candidate- and cohort-separated metrics, 4,096 matrix bootstraps, 512 development and evaluation permutations, suffix and history controls, five figures, full provenance and hashes under `S19/loops/L33`.
- **Validation:** exact replay of 280 states, molecular clocks, catalytic matrices, target basins and H32/H8 responses; CPU-float64 feature and model replay; molecule-permutation and target-invariance fixtures; suffix invariance; immutable prior, seed, storage, regeneration and artifact gates.
- **Recommended next action:** `PERMUTATION_INVARIANT_FULL_STATE_GRAPH_COORDINATE`.

## Lay summary

L31 proved that repeated short simulations from the same state have a reproducible probability of reaching the retrospectively defined basin. L33 asks whether that probability is visible without rerunning the simulator: it compresses the latest eight observed states into mass/phase, ordinary composition, and exact local-reaction-generator summaries. The primary coordinate is mathematically invariant to the completed-run target centroid; a separate target-conditioned coordinate is retained only as a retrospective oracle diagnostic.

## Frozen question

Can a fixed, basin-blind, low-dimensional history of exact GARD generator activity and simulator phase recover both the H32 committor and its H8 mediator on two unchanged evaluation cohorts and in both simulator candidates, beyond time/phase, exact-H, ordinary-path and completed-target-geometry controls?

## Inputs and methods

- 200 L28 states (development and validation) and 80 previously untouched L31 confirmation states.
- H32 responses use 128 independent branches per state; H8 is a response-only diagnostic from 64 branches.
- Eight selected-clock observations ending at the current at-risk state.
- Three prospectively fixed views: 15 phase-memory features, 35 primary basin-blind phase/composition/generator features, and a 51-feature target-conditioned oracle diagnostic.
- Endpoint, temporal slope and phase means only; no feature, window, regularization or model search.
- Exact L29 standardized L2 aggregated-binomial logistic coordinate (`C=0.1`) fit only on L28 development H32 counts.

## Evaluation metrics

| variant   | evaluationCohort   | candidateId       | modelId                            |   states |   spearmanH32 |   spearmanH8 |   h32BrierPerBranch |   h8BrierPerBranch |   h32BinomialLogLossPerBranch |   calibrationInterceptH32 |   calibrationSlopeH32 |
|:----------|:-------------------|:------------------|:-----------------------------------|---------:|--------------:|-------------:|--------------------:|-------------------:|------------------------------:|--------------------------:|----------------------:|
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-02 | BASIN_BLIND_OPERATOR_MEMORY        |       50 |     0.371317  |    0.231889  |            0.272735 |          0.304728  |                      1.14251  |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-02 | FROZEN_EXACT_H_TUBE                |       50 |     0.0832412 |   -0.0435775 |            0.256209 |          0.214585  |                      1.36535  |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-02 | FROZEN_LANDMARK_PRIOR              |       50 |     0.238344  |    0.156715  |            0.188344 |          0.105988  |                      0.563766 |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-02 | FROZEN_ORDINARY_TUBE               |       50 |    -0.0280675 |    0.0124817 |            0.302342 |          0.222933  |                      1.42135  |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-02 | FROZEN_TARGET_GEOMETRY             |       50 |     0.513337  |    0.550499  |            0.177961 |          0.150738  |                      0.544779 |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-02 | PHASE_MEMORY                       |       50 |     0.189888  |    0.0399415 |            0.246356 |          0.226386  |                      0.713472 |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-02 | TARGET_CONDITIONED_OPERATOR_MEMORY |       50 |     0.56928   |    0.631413  |            0.18175  |          0.225063  |                      0.967134 |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-03 | BASIN_BLIND_OPERATOR_MEMORY        |       50 |     0.293059  |    0.461862  |            0.177072 |          0.117059  |                      0.587713 |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-03 | FROZEN_EXACT_H_TUBE                |       50 |     0.153598  |    0.186771  |            0.257076 |          0.218984  |                      1.25286  |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-03 | FROZEN_LANDMARK_PRIOR              |       50 |     0.26106   |    0.27684   |            0.187737 |          0.11502   |                      0.559352 |                -0.675771  |             0.358035  |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-03 | FROZEN_ORDINARY_TUBE               |       50 |     0.0537162 |    0.152296  |            0.318832 |          0.27716   |                      1.41001  |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-03 | FROZEN_TARGET_GEOMETRY             |       50 |     0.654453  |    0.614099  |            0.149441 |          0.110758  |                      0.473713 |                -0.390311  |             0.584306  |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-03 | PHASE_MEMORY                       |       50 |     0.248624  |    0.301689  |            0.197131 |          0.128258  |                      0.612565 |               nan         |           nan         |
| ORIGINAL  | L28_VALIDATION     | S12F-CANDIDATE-03 | TARGET_CONDITIONED_OPERATOR_MEMORY |       50 |     0.502489  |    0.490296  |            0.166949 |          0.142226  |                      0.575609 |               nan         |           nan         |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-02 | BASIN_BLIND_OPERATOR_MEMORY        |       40 |     0.0962091 |    0.0753337 |            0.275776 |          0.242771  |                      0.909974 |               nan         |           nan         |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-02 | FROZEN_EXACT_H_TUBE                |       40 |    -0.267769  |   -0.222399  |            0.442447 |          0.335644  |                      2.75629  |               nan         |           nan         |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-02 | FROZEN_LANDMARK_PRIOR              |       40 |     0.251541  |    0.23773   |            0.185194 |          0.0872211 |                      0.555385 |               nan         |           nan         |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-02 | FROZEN_ORDINARY_TUBE               |       40 |     0.0256495 |   -0.0450767 |            0.267848 |          0.112986  |                      1.29897  |               nan         |           nan         |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-02 | FROZEN_TARGET_GEOMETRY             |       40 |     0.625265  |    0.486581  |            0.186494 |          0.118431  |                      0.565548 |                -0.538017  |             0.436327  |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-02 | PHASE_MEMORY                       |       40 |     0.182929  |    0.120925  |            0.241286 |          0.177882  |                      0.706688 |               nan         |           nan         |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-02 | TARGET_CONDITIONED_OPERATOR_MEMORY |       40 |     0.448537  |    0.445004  |            0.225149 |          0.147243  |                      0.745365 |                -0.780363  |             0.197936  |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-03 | BASIN_BLIND_OPERATOR_MEMORY        |       40 |     0.644856  |    0.608355  |            0.191019 |          0.158843  |                      0.601668 |                -0.13997   |             0.477348  |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-03 | FROZEN_EXACT_H_TUBE                |       40 |     0.0873076 |    0.0834077 |            0.329669 |          0.295672  |                      1.26454  |                -0.626032  |             0.0796867 |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-03 | FROZEN_LANDMARK_PRIOR              |       40 |     0.183141  |    0.318278  |            0.224061 |          0.132461  |                      0.661753 |                 0.008311  |             0.504022  |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-03 | FROZEN_ORDINARY_TUBE               |       40 |     0         |   -0.0226243 |            0.28741  |          0.219585  |                      1.33148  |               nan         |           nan         |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-03 | FROZEN_TARGET_GEOMETRY             |       40 |     0.544029  |    0.544027  |            0.20297  |          0.115236  |                      0.611269 |               nan         |           nan         |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-03 | PHASE_MEMORY                       |       40 |     0.225122  |    0.276184  |            0.218333 |          0.150016  |                      0.676204 |               nan         |           nan         |
| ORIGINAL  | L31_CONFIRMATION   | S12F-CANDIDATE-03 | TARGET_CONDITIONED_OPERATOR_MEMORY |       40 |     0.561115  |    0.574053  |            0.159819 |          0.0841223 |                      0.568543 |                 0.0855874 |             0.502226  |

## Locked solution gates

| evaluationCohort   | candidateId       |   states |   primarySpearmanH32 |   primarySpearmanH32Lower95 |   primarySpearmanH8 |   primarySpearmanH8Lower95 |   brierImprovementLowerVsFROZEN_LANDMARK_PRIOR |   brierImprovementLowerVsFROZEN_TARGET_GEOMETRY |   brierImprovementLowerVsFROZEN_EXACT_H_TUBE |   brierImprovementLowerVsFROZEN_ORDINARY_TUBE |   brierImprovementLowerVsPHASE_MEMORY |   developmentPermutationP |   evaluationPermutationP |   historyReversalSpearmanH32 | h32RankPassed   | h8RankPassed   | incrementalBrierPassed   | developmentPermutationPassed   | evaluationPermutationPassed   | historyReversalPassed   | suffixPassed   | cohortCandidateGatePassed   |
|:-------------------|:------------------|---------:|---------------------:|----------------------------:|--------------------:|---------------------------:|-----------------------------------------------:|------------------------------------------------:|---------------------------------------------:|----------------------------------------------:|--------------------------------------:|--------------------------:|-------------------------:|-----------------------------:|:----------------|:---------------|:-------------------------|:-------------------------------|:------------------------------|:------------------------|:---------------|:----------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 |       50 |            0.371317  |                  0.132955   |           0.231889  |                 -0.0301262 |                                     -0.17484   |                                      -0.179297  |                                   -0.118103  |                                    -0.0692122 |                            -0.0907575 |                0.218324   |               0.173489   |                    0.473242  | False           | False          | False                    | False                          | False                         | False                   | True           | False                       |
| L28_VALIDATION     | S12F-CANDIDATE-03 |       50 |            0.293059  |                  0.00101393 |           0.461862  |                  0.222771  |                                     -0.0316666 |                                      -0.0696174 |                                    0.0102739 |                                     0.0637251 |                            -0.0209651 |                0.423002   |               0.391813   |                    0.275121  | False           | False          | False                    | False                          | False                         | True                    | True           | False                       |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |       40 |            0.0962091 |                 -0.262992   |           0.0753337 |                 -0.310613  |                                     -0.165758  |                                      -0.16032   |                                    0.053186  |                                    -0.105521  |                            -0.103683  |                0.929825   |               0.962963   |                    0.0357026 | False           | False          | False                    | False                          | False                         | True                    | True           | False                       |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |       40 |            0.644856  |                  0.418131   |           0.608355  |                  0.369217  |                                     -0.029007  |                                      -0.0584845 |                                    0.0387071 |                                     0.0115741 |                            -0.034244  |                0.00389864 |               0.00194932 |                    0.424427  | True            | True           | False                    | True                           | True                          | True                    | True           | False                       |

## Source grounding and scope

The exact operator is inherited from the frozen source-defined GARD implementation and analytic moment code. General committor-learning and path-sampling literature motivated testing a low-dimensional state coordinate and held-out shooting validation; it does not identify these GARD features or the paper authors' implementation. The response basin itself remains a completed-run, matrix-specific reconstruction.

## Interpretation boundary

The primary predictor uses no branch-derived value, completed-run centroid, suffix observation, emergence estimate, intervention outcome or paper-directed threshold. H8 and H32 enter only as response variables. A passing result would establish a deterministic past-only coordinate *within the retrospective-basin-conditioned simulation task*; it would not establish the paper's exact replicator definition, early warning in empirical biology, causal emergence, or causal control. The target-conditioned view is an oracle diagnostic and is excluded from the solution gate.

## Runtime and provenance

- Repository lock: `a2e4cbdf2525fffb2a0da87b19599b62bad6c033`.
- CPU float64; one numerical-library thread; no GPU.
- Wall seconds: `220.883`; controller CPU hours: `0.061342`.

## Autonomous boundary

L33 is frozen. S20, E02, author contact, interventions, reactive-current analysis and report-bundle generation remain inactive.
