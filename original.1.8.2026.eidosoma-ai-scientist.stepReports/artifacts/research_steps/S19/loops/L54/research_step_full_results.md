# S19-L54 Full Results — Untouched Past-Observable Process-Risk Confirmation

## Top summary

- **Research step:** `E01-S19-L54-UNTOUCHED-PAST-OBSERVABLE-PROCESS-RISK-CONFIRMATION-v1.0.0`
- **Completion status:** complete; new seed-firewalled confirmation cohort frozen and exactly regenerated
- **Artifacts written:** 40 new shared catalytic matrices and initial states, 80 primary trajectories, 400 preregistered fission-landmark states, 25,600 branch futures plus an exact second campaign, unchanged L53 model replay, process-risk and realized-path outcomes, 4,096 matrix bootstraps, 512 whole-matrix permutations, seven figures, validation records, report, and hashes
- **Validation:** PASS — immutable S01–L53 baseline; 12/12 fixtures; zero seed/matrix/initial-state overlap; exact old L53 transformation/model/prediction replay; exact primary/regeneration trajectory, state, feature, prediction, branch, table, and report replay; no replacement; scope, runtime, storage, and artifact checks
- **Outcome classification:** `UNTOUCHED_PAST_OBSERVABLE_PROCESS_RISK_COORDINATE_CONFIRMED`, `PLASTIC_HEREDITY_SWITCHING_PROPENSITY_PREDICTABLE`, `SIMULATOR_PROCESS_EARLY_WARNING_CONFIRMED`, `NOT_PAPER_REPLICATION`
- **Lay summary:** The completely frozen L53 coordinate passed every untouched gate in both simulator candidates. The target is not arrival at one privileged composition. It is the probability that ordinary parent/daughter heredity breaks and a new three-fission hereditary episode forms within twelve fissions—an operational plastic-heredity switching event.
- **Recommended next action:** `HUMAN_REVIEW_CONFIRMED_SOLUTION_STOP`. The autonomous sequence stops early for mandatory human review because the preregistered confirmation succeeded.

## Frozen question and chronology

L54 asks whether a target-blind current-state/catalytic-network representation plus nine directly observed history/phase coordinates predicts an independently shot F12 break-plus-run-3 probability on wholly new matrices. The L53 PCA transforms, ridge coefficients, priors, target, threshold, landmarks, branch budget, candidates, and gates were not refit or selected from L54. Prospective prediction values were frozen before either the main realized-future labels or the independent branch outcomes were opened.

Break probability, conditional resumption after a break, and their joint event remain distinct estimands. This avoids interpreting high ordinary inheritance frequency as homeostatic recovery or fixed-attractor arrival.

## Process probabilities by horizon

| candidateId       |   horizon | targetType       |   matrices |   states |     meanQ |       sdQ |   observedRate |
|:------------------|----------:|:-----------------|-----------:|---------:|----------:|----------:|---------------:|
| S12F-CANDIDATE-02 |         4 | BREAK            |         40 |      200 | 0.2689231 | 0.2745375 |    nan         |
| S12F-CANDIDATE-02 |         4 | JOINT_BREAK_RUN3 |         40 |      200 | 0.0586154 | 0.0730838 |      0.0350000 |
| S12F-CANDIDATE-02 |         4 | RUN3_GIVEN_BREAK |         40 |      200 | 0.2812176 | 0.1805167 |    nan         |
| S12F-CANDIDATE-02 |         8 | BREAK            |         40 |      200 | 0.3798462 | 0.3190982 |    nan         |
| S12F-CANDIDATE-02 |         8 | JOINT_BREAK_RUN3 |         40 |      200 | 0.2345385 | 0.2115063 |      0.2400000 |
| S12F-CANDIDATE-02 |         8 | RUN3_GIVEN_BREAK |         40 |      200 | 0.5603159 | 0.1729218 |    nan         |
| S12F-CANDIDATE-02 |        12 | BREAK            |         40 |      200 | 0.4616154 | 0.3434499 |    nan         |
| S12F-CANDIDATE-02 |        12 | JOINT_BREAK_RUN3 |         40 |      200 | 0.3565385 | 0.2825732 |      0.3800000 |
| S12F-CANDIDATE-02 |        12 | RUN3_GIVEN_BREAK |         40 |      200 | 0.6966307 | 0.1455208 |    nan         |
| S12F-CANDIDATE-03 |         4 | BREAK            |         40 |      200 | 0.2674615 | 0.2437707 |    nan         |
| S12F-CANDIDATE-03 |         4 | JOINT_BREAK_RUN3 |         40 |      200 | 0.0543846 | 0.0573233 |      0.0700000 |
| S12F-CANDIDATE-03 |         4 | RUN3_GIVEN_BREAK |         40 |      200 | 0.2639660 | 0.1697285 |    nan         |
| S12F-CANDIDATE-03 |         8 | BREAK            |         40 |      200 | 0.4027692 | 0.2990704 |    nan         |
| S12F-CANDIDATE-03 |         8 | JOINT_BREAK_RUN3 |         40 |      200 | 0.2382308 | 0.1902908 |      0.2700000 |
| S12F-CANDIDATE-03 |         8 | RUN3_GIVEN_BREAK |         40 |      200 | 0.5608868 | 0.1546499 |    nan         |
| S12F-CANDIDATE-03 |        12 | BREAK            |         40 |      200 | 0.4853077 | 0.3263289 |    nan         |
| S12F-CANDIDATE-03 |        12 | JOINT_BREAK_RUN3 |         40 |      200 | 0.3802308 | 0.2693809 |      0.4150000 |
| S12F-CANDIDATE-03 |        12 | RUN3_GIVEN_BREAK |         40 |      200 | 0.7221498 | 0.1557169 |    nan         |

## Independent branch-half reliability

| candidateId       |   matrices |   states |   intermediateProbabilityStates |   splitHalfSpearman |   splitHalfSpearmanLower95 |   splitHalfSpearmanUpper95 |   centeredSplitHalfSpearman |   centeredSplitHalfSpearmanLower95 |   centeredSplitHalfSpearmanUpper95 | reliabilityGatePassed   |
|:------------------|-----------:|---------:|--------------------------------:|--------------------:|---------------------------:|---------------------------:|----------------------------:|-----------------------------------:|-----------------------------------:|:------------------------|
| S12F-CANDIDATE-02 |         40 |      200 |                             138 |           0.9376403 |                  0.9028121 |                  0.9490298 |                   0.6251880 |                          0.4558867 |                          0.7515855 | True                    |
| S12F-CANDIDATE-03 |         40 |      200 |                             149 |           0.9236663 |                  0.8724879 |                  0.9447903 |                   0.6061977 |                          0.4746598 |                          0.7204144 | True                    |

## F12 joint-event predictive metrics

| candidateId       | direction   |   horizon | targetType       | modelId                  |   matrices |   states |   equalMatrixMeanBranchLogLoss |   equalMatrixMeanQBrier |   qSpearman |   centeredQSpearman |   meanPredictedProbability |   meanEmpiricalQ |
|:------------------|:------------|----------:|:-----------------|:-------------------------|-----------:|---------:|-------------------------------:|------------------------:|------------:|--------------------:|---------------------------:|-----------------:|
| S12F-CANDIDATE-02 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | BETA_STRUCTURE           |         40 |      200 |                      0.6917705 |               0.0961067 |   0.0189748 |           0.0196991 |                  0.4327714 |        0.3622727 |
| S12F-CANDIDATE-02 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | DIRECT_HISTORY_PHASE     |         40 |      200 |                      0.5294400 |               0.0300974 |   0.8032901 |           0.1984271 |                  0.3821041 |        0.3622727 |
| S12F-CANDIDATE-02 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | FULL_STATE_GRAPH_HISTORY |         40 |      200 |                      0.4816388 |               0.0145353 |   0.9181662 |           0.5500061 |                  0.3553018 |        0.3622727 |
| S12F-CANDIDATE-02 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | TRAINING_PRIOR           |         40 |      200 |                      0.6528533 |               0.0782283 | nan         |         nan         |                  0.3748633 |        0.3622727 |
| S12F-CANDIDATE-02 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | BETA_STRUCTURE           |         40 |      200 |                      0.6894528 |               0.1001667 |   0.0199369 |          -0.0248412 |                  0.4199033 |        0.3551515 |
| S12F-CANDIDATE-02 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | DIRECT_HISTORY_PHASE     |         40 |      200 |                      0.5102670 |               0.0276231 |   0.8223354 |           0.3055743 |                  0.3714747 |        0.3551515 |
| S12F-CANDIDATE-02 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | FULL_STATE_GRAPH_HISTORY |         40 |      200 |                      0.4687810 |               0.0151762 |   0.9102127 |           0.6090198 |                  0.3519368 |        0.3551515 |
| S12F-CANDIDATE-02 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | TRAINING_PRIOR           |         40 |      200 |                      0.6482727 |               0.0808488 | nan         |         nan         |                  0.3648649 |        0.3551515 |
| S12F-CANDIDATE-03 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | BETA_STRUCTURE           |         40 |      200 |                      0.6963480 |               0.0894105 |  -0.0500470 |          -0.0066177 |                  0.3941865 |        0.3836364 |
| S12F-CANDIDATE-03 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | DIRECT_HISTORY_PHASE     |         40 |      200 |                      0.5494264 |               0.0284567 |   0.7850311 |           0.3450746 |                  0.3716337 |        0.3836364 |
| S12F-CANDIDATE-03 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | FULL_STATE_GRAPH_HISTORY |         40 |      200 |                      0.5003484 |               0.0118721 |   0.9168614 |           0.6968304 |                  0.3818632 |        0.3836364 |
| S12F-CANDIDATE-03 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | TRAINING_PRIOR           |         40 |      200 |                      0.6641431 |               0.0740781 | nan         |         nan         |                  0.3861115 |        0.3836364 |
| S12F-CANDIDATE-03 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | BETA_STRUCTURE           |         40 |      200 |                      0.6935956 |               0.0863786 |  -0.0685459 |          -0.0094906 |                  0.4116864 |        0.3804545 |
| S12F-CANDIDATE-03 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | DIRECT_HISTORY_PHASE     |         40 |      200 |                      0.5595459 |               0.0316973 |   0.7415765 |           0.2197817 |                  0.3713733 |        0.3804545 |
| S12F-CANDIDATE-03 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | FULL_STATE_GRAPH_HISTORY |         40 |      200 |                      0.5075433 |               0.0135802 |   0.8945050 |           0.6273593 |                  0.3821742 |        0.3804545 |
| S12F-CANDIDATE-03 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | TRAINING_PRIOR           |         40 |      200 |                      0.6626406 |               0.0714718 | nan         |         nan         |                  0.3865802 |        0.3804545 |

## Registered proper-score comparisons

| candidateId       | direction   | comparisonId   | modelId                  | referenceModelId     |   logLossImprovement |   logLossImprovementLower95 |   logLossImprovementUpper95 |   qBrierImprovement |   qBrierImprovementLower95 |   qBrierImprovementUpper95 |   fractionBootstrapLogLossPositive |   fractionBootstrapQBrierPositive |
|:------------------|:------------|:---------------|:-------------------------|:---------------------|---------------------:|----------------------------:|----------------------------:|--------------------:|---------------------------:|---------------------------:|-----------------------------------:|----------------------------------:|
| S12F-CANDIDATE-02 | A_TO_B      | FULL_VS_PRIOR  | FULL_STATE_GRAPH_HISTORY | TRAINING_PRIOR       |            0.1712146 |                   0.1289827 |                   0.2161808 |           0.0636930 |                  0.0498930 |                  0.0779651 |                          1.0000000 |                         1.0000000 |
| S12F-CANDIDATE-02 | A_TO_B      | FULL_VS_DIRECT | FULL_STATE_GRAPH_HISTORY | DIRECT_HISTORY_PHASE |            0.0478012 |                   0.0326720 |                   0.0637120 |           0.0155621 |                  0.0101606 |                  0.0212075 |                          1.0000000 |                         1.0000000 |
| S12F-CANDIDATE-02 | B_TO_A      | FULL_VS_PRIOR  | FULL_STATE_GRAPH_HISTORY | TRAINING_PRIOR       |            0.1794917 |                   0.1352506 |                   0.2252702 |           0.0656726 |                  0.0505488 |                  0.0804925 |                          1.0000000 |                         1.0000000 |
| S12F-CANDIDATE-02 | B_TO_A      | FULL_VS_DIRECT | FULL_STATE_GRAPH_HISTORY | DIRECT_HISTORY_PHASE |            0.0414860 |                   0.0259224 |                   0.0573506 |           0.0124469 |                  0.0066662 |                  0.0183781 |                          1.0000000 |                         1.0000000 |
| S12F-CANDIDATE-03 | A_TO_B      | FULL_VS_PRIOR  | FULL_STATE_GRAPH_HISTORY | TRAINING_PRIOR       |            0.1637947 |                   0.1224463 |                   0.2077968 |           0.0622060 |                  0.0488164 |                  0.0764218 |                          1.0000000 |                         1.0000000 |
| S12F-CANDIDATE-03 | A_TO_B      | FULL_VS_DIRECT | FULL_STATE_GRAPH_HISTORY | DIRECT_HISTORY_PHASE |            0.0490781 |                   0.0355115 |                   0.0636115 |           0.0165846 |                  0.0114575 |                  0.0217759 |                          1.0000000 |                         1.0000000 |
| S12F-CANDIDATE-03 | B_TO_A      | FULL_VS_PRIOR  | FULL_STATE_GRAPH_HISTORY | TRAINING_PRIOR       |            0.1550974 |                   0.1119721 |                   0.2017937 |           0.0578917 |                  0.0440248 |                  0.0727044 |                          1.0000000 |                         1.0000000 |
| S12F-CANDIDATE-03 | B_TO_A      | FULL_VS_DIRECT | FULL_STATE_GRAPH_HISTORY | DIRECT_HISTORY_PHASE |            0.0520026 |                   0.0403741 |                   0.0639672 |           0.0181171 |                  0.0133957 |                  0.0232058 |                          1.0000000 |                         1.0000000 |

## Overall and within-matrix ranks

| candidateId       | direction   | modelId                  |   qSpearman |   qSpearmanLower95 |   qSpearmanUpper95 |   centeredQSpearman |   centeredQSpearmanLower95 |   centeredQSpearmanUpper95 |
|:------------------|:------------|:-------------------------|------------:|-------------------:|-------------------:|--------------------:|---------------------------:|---------------------------:|
| S12F-CANDIDATE-02 | A_TO_B      | TRAINING_PRIOR           | nan         |        nan         |        nan         |         nan         |                nan         |                nan         |
| S12F-CANDIDATE-02 | A_TO_B      | DIRECT_HISTORY_PHASE     |   0.8032901 |          0.7034410 |          0.8527734 |           0.1984271 |                  0.0366879 |                  0.3550595 |
| S12F-CANDIDATE-02 | A_TO_B      | BETA_STRUCTURE           |   0.0189748 |         -0.2761810 |          0.2937682 |           0.0196991 |                  0.0108206 |                  0.0419458 |
| S12F-CANDIDATE-02 | A_TO_B      | FULL_STATE_GRAPH_HISTORY |   0.9181662 |          0.8680495 |          0.9357339 |           0.5500061 |                  0.4172967 |                  0.6638472 |
| S12F-CANDIDATE-02 | B_TO_A      | TRAINING_PRIOR           | nan         |        nan         |        nan         |         nan         |                nan         |                nan         |
| S12F-CANDIDATE-02 | B_TO_A      | DIRECT_HISTORY_PHASE     |   0.8223354 |          0.7363182 |          0.8656301 |           0.3055743 |                  0.1557205 |                  0.4464627 |
| S12F-CANDIDATE-02 | B_TO_A      | BETA_STRUCTURE           |   0.0199369 |         -0.2977239 |          0.3212061 |          -0.0248412 |                 -0.0943975 |                  0.0304868 |
| S12F-CANDIDATE-02 | B_TO_A      | FULL_STATE_GRAPH_HISTORY |   0.9102127 |          0.8563016 |          0.9323269 |           0.6090198 |                  0.4886640 |                  0.7016744 |
| S12F-CANDIDATE-03 | A_TO_B      | TRAINING_PRIOR           | nan         |        nan         |        nan         |         nan         |                nan         |                nan         |
| S12F-CANDIDATE-03 | A_TO_B      | DIRECT_HISTORY_PHASE     |   0.7850311 |          0.6814071 |          0.8338145 |           0.3450746 |                  0.2116740 |                  0.4642666 |
| S12F-CANDIDATE-03 | A_TO_B      | BETA_STRUCTURE           |  -0.0500470 |         -0.2949183 |          0.1896599 |          -0.0066177 |                 -0.0333913 |                  0.0263038 |
| S12F-CANDIDATE-03 | A_TO_B      | FULL_STATE_GRAPH_HISTORY |   0.9168614 |          0.8662061 |          0.9419172 |           0.6968304 |                  0.5792194 |                  0.7849448 |
| S12F-CANDIDATE-03 | B_TO_A      | TRAINING_PRIOR           | nan         |        nan         |        nan         |         nan         |                nan         |                nan         |
| S12F-CANDIDATE-03 | B_TO_A      | DIRECT_HISTORY_PHASE     |   0.7415765 |          0.6118912 |          0.8065541 |           0.2197817 |                  0.0828249 |                  0.3470391 |
| S12F-CANDIDATE-03 | B_TO_A      | BETA_STRUCTURE           |  -0.0685459 |         -0.3294534 |          0.2015023 |          -0.0094906 |                 -0.0720924 |                  0.0333956 |
| S12F-CANDIDATE-03 | B_TO_A      | FULL_STATE_GRAPH_HISTORY |   0.8945050 |          0.8217193 |          0.9294420 |           0.6273593 |                  0.5220846 |                  0.7077152 |

## Whole-matrix permutation controls

| candidateId       | direction   | modelId                  |   observedQSpearman |   observedCenteredQSpearman |   overallUpperTailP |   centeredUpperTailP |   permutations |
|:------------------|:------------|:-------------------------|--------------------:|----------------------------:|--------------------:|---------------------:|---------------:|
| S12F-CANDIDATE-02 | A_TO_B      | FULL_STATE_GRAPH_HISTORY |           0.9181662 |                   0.5500061 |           0.0019493 |            0.0019493 |            512 |
| S12F-CANDIDATE-02 | B_TO_A      | FULL_STATE_GRAPH_HISTORY |           0.9102127 |                   0.6090198 |           0.0019493 |            0.0019493 |            512 |
| S12F-CANDIDATE-03 | A_TO_B      | FULL_STATE_GRAPH_HISTORY |           0.9168614 |                   0.6968304 |           0.0019493 |            0.0019493 |            512 |
| S12F-CANDIDATE-03 | B_TO_A      | FULL_STATE_GRAPH_HISTORY |           0.8945050 |                   0.6273593 |           0.0019493 |            0.0019493 |            512 |

## Confirmation gates

| gateId                                    | candidateId       | availabilityPassed   | reliabilityPassed   | properScorePassed   | overallRankPassed   | withinMatrixRankPassed   | permutationPassed   | replayPassed   |   minimumLogLossImprovementLower95 |   minimumQBrierImprovementLower95 |   minimumQSpearmanLower95 |   minimumCenteredQSpearmanLower95 |   maximumOverallPermutationP |   maximumCenteredPermutationP | passed   |
|:------------------------------------------|:------------------|:---------------------|:--------------------|:--------------------|:--------------------|:-------------------------|:--------------------|:---------------|-----------------------------------:|----------------------------------:|--------------------------:|----------------------------------:|-----------------------------:|------------------------------:|:---------|
| UNTOUCHED_CONFIRMATION::S12F-CANDIDATE-02 | S12F-CANDIDATE-02 | True                 | True                | True                | True                | True                     | True                | True           |                          0.0259224 |                         0.0066662 |                 0.8563016 |                         0.4172967 |                    0.0019493 |                     0.0019493 | True     |
| UNTOUCHED_CONFIRMATION::S12F-CANDIDATE-03 | S12F-CANDIDATE-03 | True                 | True                | True                | True                | True                     | True                | True           |                          0.0355115 |                         0.0114575 |                 0.8217193 |                         0.5220846 |                    0.0019493 |                     0.0019493 | True     |

## Scientific interpretation

The event is an operational regime-switching process: an inheritance break followed by formation of a new short hereditary episode. It is not exact return to the old molecular composition, and neither a run of inherited fissions nor frequent resumption alone proves error correction, an organism, or a molecular attractor. Overall ranks mix stable matrix propensity and changing state risk; the separately gated within-matrix centered ranks are what test longitudinal ordering beyond stable catalytic-matrix differences.

Even a successful result is simulator-process early warning, not replication of the paper's PhiID claim. PhiID was not computed, no intervention was run, and the historical S18 paper-facing, prediction, and causal-control verdicts remain unchanged.

## Runtime and provenance

- Repository lock: `e912ad4a1ee5c2235a9b2bdb122d24a4ca45a1a7`.
- Workers: `8` with one numerical-library thread; GPU hours: 0.
- Wall time: `3.4026` hours; conservative CPU estimate: `27.2211` hours.
- New shared matrices / trajectories / restored states: 40 / 80 / 400.
- Independent branch futures per campaign: 25600; exact branch campaigns: 2.
- Matrix bootstraps: 4096; whole-matrix permutations: 512.

## Limitations

The strict `H>0.9` inheritance process is an operational simulator construct. A run of three is short, and F12 is one registered opportunity horizon. State landmarks are post-fission and do not cover every molecular-time phase. The full-state graph representation is compact and molecule-permutation-invariant. The confirmation tests transfer to new matrices and stochastic futures under the same two reconstructed simulator candidates; it does not establish author-code identity, physical chemistry, biological heredity, causal agency, or intervention efficacy.
