# S19-L38 — Past-Only Recurrence–Inheritance Outcome Construction

## Chief/human handoff

- **Step:** `E01-S19-L38-PAST-ONLY-RECURRENCE-INHERITANCE-OUTCOME-v1.0.0`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** `RECURRENCE_INHERITANCE_OUTCOME_NOT_COMMITTOR_COMPATIBLE`, `INDEPENDENT_EVENT_TARGET_REQUIRES_REDEFINITION`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** exact prefix/state/boundary replay; exact numerical/discrete/path replay of all 53,760 frozen H32/H8 streams; split-half reliability; fixed controls; 4,096 matrix bootstraps; independent complete result regeneration; immutable/runtime/storage/artifact hashes.
- **Recommended next action:** `SUSTAINED_HOMEOSTATIC_INHERITANCE_OUTCOME_CONSTRUCTION`.

## Frozen question

Does a target that needs no completed test trajectory have a reliable state-dependent probability? The event is a selected post-fission daughter that has strict parent/daughter `H>0.9` and strict `H>0.9` to an earlier inherited selected daughter separated by at least one intervening generation. Only the observed prefix and earlier states in the same future branch are eligible references.

## Prefix geometry

| evaluationCohort   | candidateId       |   states |   meanBoundaries |   meanInheritance |   priorEventFraction |
|:-------------------|:------------------|---------:|-----------------:|------------------:|---------------------:|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 |       50 |           12.24  |          0.693395 |                 0.24 |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 |       50 |           12.38  |          0.715264 |                 0.36 |
| L28_VALIDATION     | S12F-CANDIDATE-02 |       50 |           13.34  |          0.786806 |                 0.38 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |       50 |           12.84  |          0.748945 |                 0.28 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |       40 |           13     |          0.768511 |                 0.35 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |       40 |           11.475 |          0.669281 |                 0.25 |

Prior occurrence does not make a state ineligible: L38 predicts the next recurrence/inheritance event, not the first lifetime appearance of replication.

## Committor reliability

| evaluationCohort   | candidateId       | branchFamily   | targetId                         |   states |   eligibleStates |     meanQ |   minimumQ |   maximumQ |   intermediateStateCount |   observedBetweenStateVariance |   estimatedBinomialNoiseVariance |   correctedBetweenStateVariance |   correctedVarianceLower95 |   correctedVarianceUpper95 |   splitHalfSpearman |   splitHalfLower95 |   splitHalfUpper95 | reliabilityGatePassed   |
|:-------------------|:------------------|:---------------|:---------------------------------|---------:|-----------------:|----------:|-----------:|-----------:|-------------------------:|-------------------------------:|---------------------------------:|--------------------------------:|---------------------------:|---------------------------:|--------------------:|-------------------:|-------------------:|:------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            | PAST_ONLY_RECURRENCE_INHERITANCE |       50 |               50 | 0.300938  |          0 |    1       |                       25 |                     0.108008   |                      0.000823042 |                      0.107185   |                0.0630911   |                  0.141239  |            0.964052 |           0.920143 |           0.98344  | True                    |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H8             | PAST_ONLY_RECURRENCE_INHERITANCE |       50 |               50 | 0.114375  |          0 |    0.96875 |                        6 |                     0.0718216  |                      0.000490606 |                      0.071331   |                0.0242032   |                  0.116859  |            0.853155 |           0.671078 |           0.965109 | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            | PAST_ONLY_RECURRENCE_INHERITANCE |       50 |               50 | 0.193125  |          0 |    1       |                       23 |                     0.0657337  |                      0.000719753 |                      0.065014   |                0.0257232   |                  0.105124  |            0.912623 |           0.819048 |           0.957652 | True                    |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H8             | PAST_ONLY_RECURRENCE_INHERITANCE |       50 |               50 | 0.0321875 |          0 |    0.53125 |                        5 |                     0.00794613 |                      0.000370861 |                      0.00757527 |                0.000735658 |                  0.0182145 |            0.698831 |           0.391714 |           0.892503 | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            | PAST_ONLY_RECURRENCE_INHERITANCE |       40 |               40 | 0.217188  |          0 |    1       |                       19 |                     0.078905   |                      0.000732951 |                      0.078172   |                0.0333431   |                  0.123446  |            0.949196 |           0.890346 |           0.977082 | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H8             | PAST_ONLY_RECURRENCE_INHERITANCE |       40 |               40 | 0.0554687 |          0 |    1       |                        0 |                     0.0467742  |                      0.000107732 |                      0.0466665  |                2.64411e-05 |                  0.108521  |            0.572208 |           0.156516 |           0.846907 | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            | PAST_ONLY_RECURRENCE_INHERITANCE |       40 |               40 | 0.175781  |          0 |    1       |                       12 |                     0.0795116  |                      0.000530381 |                      0.0789812  |                0.025333    |                  0.130204  |            0.858553 |           0.698573 |           0.953586 | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H8             | PAST_ONLY_RECURRENCE_INHERITANCE |       40 |               40 | 0.0835937 |          0 |    1       |                        2 |                     0.066025   |                      0.00019415  |                      0.0658309  |                0.00577446  |                  0.128205  |            0.920264 |           0.716115 |           1        | False                   |

## Short shooting, controls and past proxy

| evaluationCohort   | candidateId       | comparisonId                    | comparisonType   |   definedPairs |   pointEstimate |    lower95 |   upper95 | gatePassed   |
|:-------------------|:------------------|:--------------------------------|:-----------------|---------------:|----------------:|-----------:|----------:|:-------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | INHERITANCE_ONLY_VS_PRIMARY_H32 | RANK             |             50 |       0.717642  |  0.549038  |  0.814316 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | ORIGINAL_H8_VS_PRIMARY_H32      | RANK             |             50 |       0.391387  |  0.148233  |  0.605679 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PAST_PROXY_VS_PRIMARY_H32       | RANK             |             50 |       0.253043  | -0.0513524 |  0.517125 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PRIMARY_H8_VS_PRIMARY_H32       | RANK             |             50 |       0.703755  |  0.519678  |  0.829287 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PRIMARY_MINUS_BRANCH_ONLY_H32   | DIFFERENCE       |             50 |       0.1075    |  0.0670313 |  0.153535 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PRIMARY_MINUS_PERMUTED_H32      | DIFFERENCE       |             50 |       0.103594  |  0.0634375 |  0.148945 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PRIMARY_MINUS_UNRELATED_H32     | DIFFERENCE       |             50 |       0.107344  |  0.0658398 |  0.153008 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | INHERITANCE_ONLY_VS_PRIMARY_H32 | RANK             |             50 |       0.61789   |  0.399258  |  0.778101 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | ORIGINAL_H8_VS_PRIMARY_H32      | RANK             |             50 |       0.221134  | -0.0680164 |  0.466935 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PAST_PROXY_VS_PRIMARY_H32       | RANK             |             50 |       0.532844  |  0.286321  |  0.728009 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PRIMARY_H8_VS_PRIMARY_H32       | RANK             |             50 |       0.583843  |  0.346762  |  0.757995 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PRIMARY_MINUS_BRANCH_ONLY_H32   | DIFFERENCE       |             50 |       0.0823437 |  0.0477148 |  0.124629 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PRIMARY_MINUS_PERMUTED_H32      | DIFFERENCE       |             50 |       0.0823437 |  0.0473437 |  0.124941 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PRIMARY_MINUS_UNRELATED_H32     | DIFFERENCE       |             50 |       0.0823437 |  0.0475    |  0.121719 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | INHERITANCE_ONLY_VS_PRIMARY_H32 | RANK             |             40 |       0.623929  |  0.379032  |  0.794659 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | ORIGINAL_H8_VS_PRIMARY_H32      | RANK             |             40 |       0.269744  | -0.0576055 |  0.552422 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PAST_PROXY_VS_PRIMARY_H32       | RANK             |             40 |       0.517874  |  0.238095  |  0.738026 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PRIMARY_H8_VS_PRIMARY_H32       | RANK             |             40 |       0.634053  |  0.413785  |  0.803263 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PRIMARY_MINUS_BRANCH_ONLY_H32   | DIFFERENCE       |             40 |       0.0910156 |  0.0472656 |  0.145288 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PRIMARY_MINUS_PERMUTED_H32      | DIFFERENCE       |             40 |       0.0910156 |  0.0455078 |  0.145239 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PRIMARY_MINUS_UNRELATED_H32     | DIFFERENCE       |             40 |       0.0896484 |  0.0447998 |  0.143945 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | INHERITANCE_ONLY_VS_PRIMARY_H32 | RANK             |             40 |       0.730999  |  0.536642  |  0.844523 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | ORIGINAL_H8_VS_PRIMARY_H32      | RANK             |             40 |       0.550079  |  0.294019  |  0.753547 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PAST_PROXY_VS_PRIMARY_H32       | RANK             |             40 |       0.361378  |  0.0467301 |  0.618618 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PRIMARY_H8_VS_PRIMARY_H32       | RANK             |             40 |       0.57371   |  0.29436   |  0.737803 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PRIMARY_MINUS_BRANCH_ONLY_H32   | DIFFERENCE       |             40 |       0.0558594 |  0.0195312 |  0.108789 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PRIMARY_MINUS_PERMUTED_H32      | DIFFERENCE       |             40 |       0.0558594 |  0.0197998 |  0.106372 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PRIMARY_MINUS_UNRELATED_H32     | DIFFERENCE       |             40 |       0.0527344 |  0.0193359 |  0.101758 | True         |

## One-realization calibration diagnostic

| evaluationCohort   | candidateId       |   states |   observedEventFraction |   meanPredictedProbability |     brier |   brierLower95 |   brierUpper95 |   logLoss | evaluationOnly   |
|:-------------------|:------------------|---------:|------------------------:|---------------------------:|----------:|---------------:|---------------:|----------:|:-----------------|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 |       50 |                   0.16  |                   0.158281 | 0.0449939 |      0.0156012 |      0.081946  |  0.155261 | True             |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 |       50 |                   0.26  |                   0.161563 | 0.158306  |      0.0931299 |      0.231674  |  0.461012 | True             |
| L28_VALIDATION     | S12F-CANDIDATE-02 |       50 |                   0.28  |                   0.300938 | 0.107974  |      0.0566347 |      0.163981  |  0.326847 | True             |
| L28_VALIDATION     | S12F-CANDIDATE-03 |       50 |                   0.16  |                   0.193125 | 0.0939038 |      0.0411885 |      0.154358  |  0.310376 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |       40 |                   0.225 |                   0.217188 | 0.0819153 |      0.0345348 |      0.14308   |  0.264344 | True             |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |       40 |                   0.125 |                   0.175781 | 0.0404541 |      0.012184  |      0.0807611 |  0.146092 | True             |

The completed observed suffix is used only as one evaluation realization; it does not define the target or any branch label.

## Decision gates

| evaluationCohort   | candidateId       | primaryH32Reliable   | primaryH8Reliable   | h8ToH32RankPassed   | speciesPermutationControlPassed   | unrelatedMatrixControlPassed   | pastOnlyOutcomeCommittorPassed   | shortShootingCoordinatePassed   | pastProxyRankPassed   | originalTeacherTransferPassed   |
|:-------------------|:------------------|:---------------------|:--------------------|:--------------------|:----------------------------------|:-------------------------------|:---------------------------------|:--------------------------------|:----------------------|:--------------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | True                 | False               | True                | True                              | True                           | True                             | False                           | False                 | False                           |
| L28_VALIDATION     | S12F-CANDIDATE-03 | True                 | False               | True                | True                              | True                           | True                             | False                           | False                 | False                           |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | False                | False               | True                | True                              | True                           | False                            | False                           | False                 | False                           |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | False                | False               | False               | True                              | True                           | False                            | False                           | False                 | False                           |

## Validation and provenance

- Repository lock: `618cf0615494b3f4e0ba8f24ee6361dfc49cdd16`.
- Workers: `8` with one numerical-library thread per worker; GPU hours `0`.
- Wall time: `1334.91` seconds.
- New matrices, trajectories and branch streams: `0/0/0`.
- Frozen branch streams scored: `53760` and independently scored again for full regeneration.
- No threshold, horizon, simulator, target rule, control, candidate or state was selected after outcomes.

## Interpretation boundary

This is an adaptive target-construction audit. Even a positive result would establish only a simulator-defined probability of a future recurrence/inheritance event and, conditionally, a short stochastic-shooting estimator. It would not identify the paper label, prove a static biomarker, establish initial appearance, causal emergence, intervention efficacy, or biological causation.

## Next boundary

L38 is frozen. The standing human authorization permits `SUSTAINED_HOMEOSTATIC_INHERITANCE_OUTCOME_CONSTRUCTION` as the next bounded loop through L55. S20, E02, author contact, interventions and report generation remain inactive.
