# S19-L50 Full Results — Fission-Aligned Process-Committor Horizon and Phase Identifiability

## Top summary

- **Research step:** `E01-S19-L50-FISSION-ALIGNED-PROCESS-COMMITTOR-HORIZON-v1.0.0`
- **Completion status:** complete; additive exploratory simulator evidence
- **Artifacts written:** immutable/source/input/seed locks, 80 shared matrices, 800 exact post-fission states, 51,200 F12 branches with nested F4/F8/F12 outcomes, exact branch replay, 4,096 matrix bootstraps, 512 whole-risk-trajectory permutations per candidate/horizon, six figures, report and hash manifests
- **Validation:** PASS — immutable S01–L49R baseline; nine fixtures; zero matrix overlap with L49R; exact post-fission generation and state restoration; seed firewall; two exact branch campaigns; exact analysis/report regeneration; runtime, storage and artifact hashes
- **Outcome classification:** `FISSION_ALIGNED_JOINT_PROCESS_RISK_IDENTIFIED`, `SHOOTING_NOT_INCREMENTAL_BEYOND_DIRECT_HISTORY`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Lay summary:** L50 asks the same online process question at exact post-fission generations, so the observation clock now matches the event clock. It measures the probability of a future inheritance break followed by a new three-fission hereditary episode within four, eight and twelve future fissions, without choosing a horizon after seeing the results.
- **Recommended next action:** `L51_PROCESS_HAZARD_RENEWAL_BASELINE_AUDIT` under the autonomous authorization through L65. S20, E02, author contact and intervention work remain inactive.

## Frozen question and design

L49R mixed molecular-update and post-fission states. Its joint process probability was reliable across branch halves, but its conditional renewal component was undefined in some highly stable matrices and its full gate failed. L50 prospectively aligns every state to completed fissions 20, 35, 50, 65 and 80. It excludes all L49R-selected matrices and uses 40 new development plus 40 new validation identities from the frozen L23 cohort. The fixed F4, F8 and F12 horizons are nested prefixes of the same branch and are never selected against outcome proximity.

The outcome remains online and destination-free: observe the first future strict `H<=0.9` parent/daughter break, then certify a new episode only after three consecutive strict `H>0.9` inherited fissions. The primary probability is the joint chance of break plus certification. Break probability and certification conditional on a break are retained separately.

## Measurement reliability

| candidateId       |   horizon | targetType       |   states |   dataInformedFraction |   intermediateStates |   splitHalfSpearman |   centeredSplitHalfSpearman |   centeredSplitHalfSpearmanLower95 |   correctedWithinMatrixVariance |   correctedWithinMatrixVarianceLower95 |
|:------------------|----------:|:-----------------|---------:|-----------------------:|---------------------:|--------------------:|----------------------------:|-----------------------------------:|--------------------------------:|---------------------------------------:|
| S12F-CANDIDATE-02 |         4 | JOINT_BREAK_RUN3 |      200 |               1.000000 |            39.000000 |            0.708762 |                    0.637925 |                           0.531303 |                        0.002746 |                               0.001749 |
| S12F-CANDIDATE-02 |         4 | RUN3_GIVEN_BREAK |      200 |               0.805000 |           109.000000 |            0.394535 |                    0.330467 |                           0.193753 |                        0.007722 |                               0.003696 |
| S12F-CANDIDATE-02 |         8 | JOINT_BREAK_RUN3 |      200 |               1.000000 |           144.000000 |            0.891617 |                    0.783488 |                           0.683086 |                        0.019128 |                               0.013621 |
| S12F-CANDIDATE-02 |         8 | RUN3_GIVEN_BREAK |      200 |               0.890000 |           176.000000 |            0.502356 |                    0.481430 |                           0.344290 |                        0.024488 |                               0.016402 |
| S12F-CANDIDATE-02 |        12 | JOINT_BREAK_RUN3 |      200 |               1.000000 |           165.000000 |            0.884118 |                    0.669412 |                           0.542896 |                        0.016147 |                               0.011666 |
| S12F-CANDIDATE-02 |        12 | RUN3_GIVEN_BREAK |      200 |               0.895000 |           164.000000 |            0.438885 |                    0.404537 |                           0.220422 |                        0.010362 |                               0.007543 |
| S12F-CANDIDATE-03 |         4 | JOINT_BREAK_RUN3 |      200 |               1.000000 |            37.000000 |            0.703227 |                    0.556986 |                           0.411332 |                        0.002005 |                               0.001274 |
| S12F-CANDIDATE-03 |         4 | RUN3_GIVEN_BREAK |      200 |               0.800000 |           106.000000 |            0.413877 |                    0.294608 |                           0.120718 |                        0.007959 |                               0.003725 |
| S12F-CANDIDATE-03 |         8 | JOINT_BREAK_RUN3 |      200 |               1.000000 |           146.000000 |            0.911086 |                    0.808634 |                           0.719976 |                        0.019605 |                               0.013134 |
| S12F-CANDIDATE-03 |         8 | RUN3_GIVEN_BREAK |      200 |               0.885000 |           176.000000 |            0.519835 |                    0.491629 |                           0.335148 |                        0.016980 |                               0.010297 |
| S12F-CANDIDATE-03 |        12 | JOINT_BREAK_RUN3 |      200 |               1.000000 |           169.000000 |            0.934631 |                    0.809151 |                           0.723873 |                        0.019137 |                               0.012993 |
| S12F-CANDIDATE-03 |        12 | RUN3_GIVEN_BREAK |      200 |               0.895000 |           169.000000 |            0.556611 |                    0.498402 |                           0.313592 |                        0.009493 |                               0.004786 |

The joint probability is defined for every state. The conditional component is shown separately because a state with no break in 64 branches has no empirical conditional-renewal trials; this is physical stability information, not missing data to impute.

## Fixed-landmark risk geometry

| candidateId       |   horizon |   completedFissionLandmark |    meanQ |   medianQ |   minimumQ |   maximumQ |
|:------------------|----------:|---------------------------:|---------:|----------:|-----------:|-----------:|
| S12F-CANDIDATE-02 |         4 |                         20 | 0.057692 |  0.023077 |   0.007692 |   0.284615 |
| S12F-CANDIDATE-02 |         4 |                         35 | 0.045000 |  0.023077 |   0.007692 |   0.146154 |
| S12F-CANDIDATE-02 |         4 |                         50 | 0.056154 |  0.030769 |   0.007692 |   0.300000 |
| S12F-CANDIDATE-02 |         4 |                         65 | 0.069615 |  0.038462 |   0.007692 |   0.253846 |
| S12F-CANDIDATE-02 |         4 |                         80 | 0.056154 |  0.023077 |   0.007692 |   0.284615 |
| S12F-CANDIDATE-02 |         8 |                         20 | 0.262692 |  0.223077 |   0.007692 |   0.715385 |
| S12F-CANDIDATE-02 |         8 |                         35 | 0.235000 |  0.230769 |   0.007692 |   0.592308 |
| S12F-CANDIDATE-02 |         8 |                         50 | 0.232308 |  0.192308 |   0.007692 |   0.761538 |
| S12F-CANDIDATE-02 |         8 |                         65 | 0.276923 |  0.276923 |   0.007692 |   0.684615 |
| S12F-CANDIDATE-02 |         8 |                         80 | 0.233846 |  0.207692 |   0.007692 |   0.607692 |
| S12F-CANDIDATE-02 |        12 |                         20 | 0.419615 |  0.384615 |   0.007692 |   0.915385 |
| S12F-CANDIDATE-02 |        12 |                         35 | 0.395000 |  0.407692 |   0.007692 |   0.761538 |
| S12F-CANDIDATE-02 |        12 |                         50 | 0.385000 |  0.392308 |   0.007692 |   0.869231 |
| S12F-CANDIDATE-02 |        12 |                         65 | 0.434231 |  0.469231 |   0.007692 |   0.853846 |
| S12F-CANDIDATE-02 |        12 |                         80 | 0.400769 |  0.438462 |   0.007692 |   0.838462 |
| S12F-CANDIDATE-03 |         4 |                         20 | 0.059615 |  0.030769 |   0.007692 |   0.238462 |
| S12F-CANDIDATE-03 |         4 |                         35 | 0.056154 |  0.038462 |   0.007692 |   0.192308 |
| S12F-CANDIDATE-03 |         4 |                         50 | 0.063846 |  0.069231 |   0.007692 |   0.207692 |
| S12F-CANDIDATE-03 |         4 |                         65 | 0.047308 |  0.038462 |   0.007692 |   0.238462 |
| S12F-CANDIDATE-03 |         4 |                         80 | 0.052308 |  0.038462 |   0.007692 |   0.176923 |
| S12F-CANDIDATE-03 |         8 |                         20 | 0.267308 |  0.215385 |   0.007692 |   0.838462 |
| S12F-CANDIDATE-03 |         8 |                         35 | 0.275769 |  0.253846 |   0.007692 |   0.592308 |
| S12F-CANDIDATE-03 |         8 |                         50 | 0.275000 |  0.269231 |   0.007692 |   0.684615 |
| S12F-CANDIDATE-03 |         8 |                         65 | 0.219615 |  0.192308 |   0.007692 |   0.700000 |
| S12F-CANDIDATE-03 |         8 |                         80 | 0.242308 |  0.161538 |   0.007692 |   0.684615 |
| S12F-CANDIDATE-03 |        12 |                         20 | 0.423077 |  0.438462 |   0.007692 |   0.930769 |
| S12F-CANDIDATE-03 |        12 |                         35 | 0.436538 |  0.453846 |   0.007692 |   0.823077 |
| S12F-CANDIDATE-03 |        12 |                         50 | 0.448077 |  0.492308 |   0.007692 |   0.869231 |
| S12F-CANDIDATE-03 |        12 |                         65 | 0.384615 |  0.330769 |   0.007692 |   0.884615 |
| S12F-CANDIDATE-03 |        12 |                         80 | 0.392308 |  0.338462 |   0.007692 |   0.869231 |

## F12 independent realized-future forecast

| candidateId       | modelId            |   states |    brier |   logLoss |    auroc |    auprc |   balancedAccuracy |   positiveRate |
|:------------------|:-------------------|---------:|---------:|----------:|---------:|---------:|-------------------:|---------------:|
| S12F-CANDIDATE-02 | DEVELOPMENT_PRIOR  |      200 | 0.235631 |  0.664130 | 0.500000 | 0.380000 |           0.500000 |       0.380000 |
| S12F-CANDIDATE-02 | PAST_CONTROLS      |      200 | 0.200812 |  0.576461 | 0.731112 | 0.562544 |           0.666596 |       0.380000 |
| S12F-CANDIDATE-02 | PAST_PLUS_SHOOTING |      200 | 0.188845 |  0.549477 | 0.778544 | 0.615189 |           0.702462 |       0.380000 |
| S12F-CANDIDATE-02 | SHOOTING_Q_JOINT   |      200 | 0.187453 |  0.546098 | 0.775626 | 0.615864 |           0.667020 |       0.380000 |
| S12F-CANDIDATE-02 | TIME_ONLY          |      200 | 0.237961 |  0.669022 | 0.491511 | 0.374918 |           0.500000 |       0.380000 |
| S12F-CANDIDATE-03 | DEVELOPMENT_PRIOR  |      200 | 0.241178 |  0.675521 | 0.500000 | 0.400000 |           0.500000 |       0.400000 |
| S12F-CANDIDATE-03 | PAST_CONTROLS      |      200 | 0.208667 |  0.605345 | 0.718021 | 0.608798 |           0.643750 |       0.400000 |
| S12F-CANDIDATE-03 | PAST_PLUS_SHOOTING |      200 | 0.178540 |  0.528359 | 0.796979 | 0.722187 |           0.700000 |       0.400000 |
| S12F-CANDIDATE-03 | SHOOTING_Q_JOINT   |      200 | 0.176788 |  0.519524 | 0.801823 | 0.721717 |           0.710417 |       0.400000 |
| S12F-CANDIDATE-03 | TIME_ONLY          |      200 | 0.241284 |  0.675767 | 0.508333 | 0.403021 |           0.500000 |       0.400000 |

## Registered Brier comparisons across horizons

| candidateId       |   horizon | modelId            | referenceModelId   |   brierImprovement |   brierImprovementLower95 |   brierImprovementUpper95 |   fractionBootstrapPositive |
|:------------------|----------:|:-------------------|:-------------------|-------------------:|--------------------------:|--------------------------:|----------------------------:|
| S12F-CANDIDATE-02 |         4 | PAST_PLUS_SHOOTING | PAST_CONTROLS      |           0.001308 |                 -0.002679 |                  0.006051 |                    0.702637 |
| S12F-CANDIDATE-02 |         4 | SHOOTING_Q_JOINT   | DEVELOPMENT_PRIOR  |           0.001588 |                 -0.002998 |                  0.006932 |                    0.722656 |
| S12F-CANDIDATE-02 |         4 | SHOOTING_Q_JOINT   | PAST_CONTROLS      |           0.002860 |                 -0.002327 |                  0.008817 |                    0.836182 |
| S12F-CANDIDATE-02 |         8 | PAST_PLUS_SHOOTING | PAST_CONTROLS      |           0.010672 |                 -0.006781 |                  0.027104 |                    0.892822 |
| S12F-CANDIDATE-02 |         8 | SHOOTING_Q_JOINT   | DEVELOPMENT_PRIOR  |           0.027581 |                  0.006829 |                  0.048845 |                    0.997070 |
| S12F-CANDIDATE-02 |         8 | SHOOTING_Q_JOINT   | PAST_CONTROLS      |           0.011339 |                 -0.004425 |                  0.025667 |                    0.929688 |
| S12F-CANDIDATE-02 |        12 | PAST_PLUS_SHOOTING | PAST_CONTROLS      |           0.011967 |                 -0.005169 |                  0.028809 |                    0.917725 |
| S12F-CANDIDATE-02 |        12 | SHOOTING_Q_JOINT   | DEVELOPMENT_PRIOR  |           0.048178 |                  0.023472 |                  0.070573 |                    1.000000 |
| S12F-CANDIDATE-02 |        12 | SHOOTING_Q_JOINT   | PAST_CONTROLS      |           0.013359 |                 -0.006152 |                  0.032697 |                    0.913086 |
| S12F-CANDIDATE-03 |         4 | PAST_PLUS_SHOOTING | PAST_CONTROLS      |          -0.000649 |                 -0.003660 |                  0.002842 |                    0.336182 |
| S12F-CANDIDATE-03 |         4 | SHOOTING_Q_JOINT   | DEVELOPMENT_PRIOR  |           0.005254 |                 -0.000181 |                  0.011699 |                    0.968018 |
| S12F-CANDIDATE-03 |         4 | SHOOTING_Q_JOINT   | PAST_CONTROLS      |           0.001786 |                 -0.001894 |                  0.006019 |                    0.805908 |
| S12F-CANDIDATE-03 |         8 | PAST_PLUS_SHOOTING | PAST_CONTROLS      |           0.017047 |                  0.001838 |                  0.032775 |                    0.987061 |
| S12F-CANDIDATE-03 |         8 | SHOOTING_Q_JOINT   | DEVELOPMENT_PRIOR  |           0.038215 |                  0.020584 |                  0.055690 |                    1.000000 |
| S12F-CANDIDATE-03 |         8 | SHOOTING_Q_JOINT   | PAST_CONTROLS      |           0.021910 |                  0.006401 |                  0.038095 |                    0.997314 |
| S12F-CANDIDATE-03 |        12 | PAST_PLUS_SHOOTING | PAST_CONTROLS      |           0.030127 |                  0.009336 |                  0.050914 |                    0.997070 |
| S12F-CANDIDATE-03 |        12 | SHOOTING_Q_JOINT   | DEVELOPMENT_PRIOR  |           0.064390 |                  0.038308 |                  0.090227 |                    1.000000 |
| S12F-CANDIDATE-03 |        12 | SHOOTING_Q_JOINT   | PAST_CONTROLS      |           0.031879 |                  0.010051 |                  0.053751 |                    0.997070 |

Past controls are fitted only on development matrices and contain generation, full and recent inheritance frequency, trailing streak, latest parent/daughter H and fissions since the latest break. Whole five-state q trajectories are permuted among validation matrices as the registered alignment null.

## Scientific gates

| gateId                             | candidateId       |   horizon | gateFamily                  |   centeredSplitHalfLower95 |   correctedWithinVarianceLower95 |   intermediateStates |   observedJointEvents |   brierImprovementOverPriorLower95 |   brierImprovementOverControlsLower95 |   permutationP | passed   |
|:-----------------------------------|:------------------|----------:|:----------------------------|---------------------------:|---------------------------------:|---------------------:|----------------------:|-----------------------------------:|--------------------------------------:|---------------:|:---------|
| MEASUREMENT_H4::S12F-CANDIDATE-02  | S12F-CANDIDATE-02 |         4 | JOINT_PROCESS_MEASUREMENT   |                   0.531303 |                         0.001749 |            39.000000 |                    10 |                         nan        |                            nan        |     nan        | True     |
| FORECAST_H4::S12F-CANDIDATE-02     | S12F-CANDIDATE-02 |         4 | INDEPENDENT_OBSERVED_FUTURE |                 nan        |                       nan        |           nan        |                    10 |                          -0.002998 |                             -0.002327 |       0.007797 | False    |
| MEASUREMENT_H8::S12F-CANDIDATE-02  | S12F-CANDIDATE-02 |         8 | JOINT_PROCESS_MEASUREMENT   |                   0.683086 |                         0.013621 |           144.000000 |                    50 |                         nan        |                            nan        |     nan        | True     |
| FORECAST_H8::S12F-CANDIDATE-02     | S12F-CANDIDATE-02 |         8 | INDEPENDENT_OBSERVED_FUTURE |                 nan        |                       nan        |           nan        |                    50 |                           0.006829 |                             -0.004425 |       0.001949 | False    |
| MEASUREMENT_H12::S12F-CANDIDATE-02 | S12F-CANDIDATE-02 |        12 | JOINT_PROCESS_MEASUREMENT   |                   0.542896 |                         0.011666 |           165.000000 |                    76 |                         nan        |                            nan        |     nan        | True     |
| FORECAST_H12::S12F-CANDIDATE-02    | S12F-CANDIDATE-02 |        12 | INDEPENDENT_OBSERVED_FUTURE |                 nan        |                       nan        |           nan        |                    76 |                           0.023472 |                             -0.006152 |       0.001949 | False    |
| MEASUREMENT_H4::S12F-CANDIDATE-03  | S12F-CANDIDATE-03 |         4 | JOINT_PROCESS_MEASUREMENT   |                   0.411332 |                         0.001274 |            37.000000 |                    10 |                         nan        |                            nan        |     nan        | True     |
| FORECAST_H4::S12F-CANDIDATE-03     | S12F-CANDIDATE-03 |         4 | INDEPENDENT_OBSERVED_FUTURE |                 nan        |                       nan        |           nan        |                    10 |                          -0.000181 |                             -0.001894 |       0.001949 | False    |
| MEASUREMENT_H8::S12F-CANDIDATE-03  | S12F-CANDIDATE-03 |         8 | JOINT_PROCESS_MEASUREMENT   |                   0.719976 |                         0.013134 |           146.000000 |                    49 |                         nan        |                            nan        |     nan        | True     |
| FORECAST_H8::S12F-CANDIDATE-03     | S12F-CANDIDATE-03 |         8 | INDEPENDENT_OBSERVED_FUTURE |                 nan        |                       nan        |           nan        |                    49 |                           0.020584 |                              0.006401 |       0.001949 | True     |
| MEASUREMENT_H12::S12F-CANDIDATE-03 | S12F-CANDIDATE-03 |        12 | JOINT_PROCESS_MEASUREMENT   |                   0.723873 |                         0.012993 |           169.000000 |                    80 |                         nan        |                            nan        |     nan        | True     |
| FORECAST_H12::S12F-CANDIDATE-03    | S12F-CANDIDATE-03 |        12 | INDEPENDENT_OBSERVED_FUTURE |                 nan        |                       nan        |           nan        |                    80 |                           0.038308 |                              0.010051 |       0.001949 | True     |
| COMPLETE_PRIMARY_F12               | BOTH              |        12 | COMPLETE                    |                 nan        |                       nan        |           nan        |                   156 |                         nan        |                            nan        |     nan        | False    |

## Interpretation boundary

A passing result supports only an online-defined, simulator-accessible process propensity. It does not establish a static biomarker, paper replication, causal-emergence mechanism, intervention effect or real-chemistry claim. F4/F8 results cannot replace the registered F12 primary gate. Any favorable result remains adaptive and requires a later untouched matrix confirmation.

## Runtime and provenance

- Repository lock: `3b932a189665ad3b9c0096f98c36ae1dfa1527da`.
- Workers: `8` with one numerical-library thread per worker; GPU hours: 0.
- Wall time: `177.101` minutes; worker CPU upper estimate: `0.736974` hours.
- New primary matrices/trajectories: 0/0; new branch streams: `51,200`; exact branch campaigns: 2.
- Matrix bootstraps: 4096; matrix-trajectory permutations: 512 per candidate/horizon.

## Limitations

The 80 matrices come from a previously generated but L50-outcome-unopened L23 cohort, so the loop is exploratory rather than a new-matrix confirmation. Five fixed generations cannot describe every regime switch. Branch estimates remain Monte Carlo measurements. The process detects renewed local compositional heredity, not restoration of an old composition or function. Repeated states within a catalytic matrix are dependent and are therefore resampled and permuted only as whole matrix trajectories.
