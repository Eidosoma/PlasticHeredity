# S19-L53 Full Results — Past-Observable Regime-Capacity Proxy

## Top summary

- **Research step:** `E01-S19-L53-PAST-OBSERVABLE-REGIME-CAPACITY-PROXY-v1.0.0`
- **Completion status:** complete; additive adaptive exploratory analysis-only evidence
- **Artifacts written:** exact 800-state/beta replay, fixed direct-history, beta-only and target-blind full-state graph features, development-only PCA/ridge models, A-to-B/B-to-A branch-half scoring, F4/F8/F12 break/resumption/joint results, 4,096 matrix bootstraps, 512 whole-matrix permutations, feature attributions, six figures, report and hash manifests
- **Validation:** PASS — immutable S01–L52 baseline; ten fixtures; exact L50 state/outcome replay; exact beta hashes and matrix-constant beta signatures; target-blind graph invariance; development/validation and branch-half separation; two exact feature/model/table passes; runtime, storage and artifact hashes
- **Outcome classification:** `BETA_STRUCTURE_DOES_NOT_EXPLAIN_TRANSFERABLE_REGIME_CAPACITY`, `PAST_OBSERVABLE_STATE_LOCAL_HAZARD_PROXY_IDENTIFIED`, `PAST_OBSERVABLE_PROCESS_RISK_LEAD`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Lay summary:** L52 showed that a few simulated futures reveal a reliable local switching law. L53 asks whether that law can instead be inferred from what is already visible: recent heredity, the catalytic network itself, or the complete present physical state. Stable matrix capacity and changing within-matrix warning are adjudicated separately.
- **Recommended next action:** `L54_UNTOUCHED_PAST_OBSERVABLE_PROCESS_RISK_CONFIRMATION` under the bounded autonomous authorization through L65. No L54 work occurs inside L53; S20, E02, author contact, Phi and interventions remain inactive.

## Frozen design

The strict parent/daughter `H>0.9` process, F4/F8/F12 horizons, break, conditional run-3 resumption and joint event are unchanged. Models fit only development matrices with branch half A or B and score validation matrices with the opposite half. The beta-only signature is constant across the five states of one matrix and can support only a stable-capacity interpretation. The full graph is the frozen L34 target-blind representation; no graph layer, feature subset, PCA dimension or regularization value was searched.

## Primary F12 joint-event metrics

| candidateId       | direction   |   horizon | targetType       | modelId                  |   matrices |   states |   equalMatrixMeanBranchLogLoss |   equalMatrixMeanQBrier |   qSpearman |   centeredQSpearman |   meanPredictedProbability |   meanEmpiricalQ |
|:------------------|:------------|----------:|:-----------------|:-------------------------|-----------:|---------:|-------------------------------:|------------------------:|------------:|--------------------:|---------------------------:|-----------------:|
| S12F-CANDIDATE-02 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | BETA_STRUCTURE           |         40 |      200 |                      0.6875456 |               0.0663846 |   0.2550986 |           0.0259431 |                  0.3677626 |        0.4122727 |
| S12F-CANDIDATE-02 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | DIRECT_HISTORY_PHASE     |         40 |      200 |                      0.5938071 |               0.0282120 |   0.7353013 |           0.4183021 |                  0.4088606 |        0.4122727 |
| S12F-CANDIDATE-02 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | FULL_STATE_GRAPH_HISTORY |         40 |      200 |                      0.5569532 |               0.0162152 |   0.8554559 |           0.6881931 |                  0.3969798 |        0.4122727 |
| S12F-CANDIDATE-02 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | TRAINING_PRIOR           |         40 |      200 |                      0.6792229 |               0.0627989 | nan         |         nan         |                  0.3748633 |        0.4122727 |
| S12F-CANDIDATE-02 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | BETA_STRUCTURE           |         40 |      200 |                      0.6859707 |               0.0649822 |   0.2619887 |           0.0273636 |                  0.3633579 |        0.4043939 |
| S12F-CANDIDATE-02 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | DIRECT_HISTORY_PHASE     |         40 |      200 |                      0.5961287 |               0.0281576 |   0.7335984 |           0.3782808 |                  0.3978436 |        0.4043939 |
| S12F-CANDIDATE-02 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | FULL_STATE_GRAPH_HISTORY |         40 |      200 |                      0.5615323 |               0.0172889 |   0.8474784 |           0.6650280 |                  0.3854954 |        0.4043939 |
| S12F-CANDIDATE-02 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | TRAINING_PRIOR           |         40 |      200 |                      0.6764213 |               0.0611961 | nan         |         nan         |                  0.3648649 |        0.4043939 |
| S12F-CANDIDATE-03 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | BETA_STRUCTURE           |         40 |      200 |                      0.7042815 |               0.0763106 |   0.0556491 |           0.0041556 |                  0.3780415 |        0.4195455 |
| S12F-CANDIDATE-03 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | DIRECT_HISTORY_PHASE     |         40 |      200 |                      0.5905597 |               0.0291609 |   0.7269512 |           0.4575992 |                  0.4018787 |        0.4195455 |
| S12F-CANDIDATE-03 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | FULL_STATE_GRAPH_HISTORY |         40 |      200 |                      0.5492566 |               0.0147890 |   0.8746284 |           0.7154814 |                  0.4076766 |        0.4195455 |
| S12F-CANDIDATE-03 | A_TO_B      |        12 | JOINT_BREAK_RUN3 | TRAINING_PRIOR           |         40 |      200 |                      0.6813140 |               0.0659626 | nan         |         nan         |                  0.3861115 |        0.4195455 |
| S12F-CANDIDATE-03 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | BETA_STRUCTURE           |         40 |      200 |                      0.7081934 |               0.0796537 |  -0.0153962 |           0.0084089 |                  0.3936657 |        0.4168182 |
| S12F-CANDIDATE-03 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | DIRECT_HISTORY_PHASE     |         40 |      200 |                      0.5941649 |               0.0326503 |   0.6984221 |           0.3897785 |                  0.4029284 |        0.4168182 |
| S12F-CANDIDATE-03 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | FULL_STATE_GRAPH_HISTORY |         40 |      200 |                      0.5494829 |               0.0163691 |   0.8625301 |           0.7091934 |                  0.4145754 |        0.4168182 |
| S12F-CANDIDATE-03 | B_TO_A      |        12 | JOINT_BREAK_RUN3 | TRAINING_PRIOR           |         40 |      200 |                      0.6799548 |               0.0667270 | nan         |         nan         |                  0.3865802 |        0.4168182 |

## Registered proper-score comparisons

| candidateId       | direction   | comparisonId    | modelId                  | referenceModelId     |   logLossImprovement |   logLossImprovementLower95 |   logLossImprovementUpper95 |   fractionBootstrapPositive |
|:------------------|:------------|:----------------|:-------------------------|:---------------------|---------------------:|----------------------------:|----------------------------:|----------------------------:|
| S12F-CANDIDATE-02 | A_TO_B      | DIRECT_VS_PRIOR | DIRECT_HISTORY_PHASE     | TRAINING_PRIOR       |            0.0854158 |                   0.0513367 |                   0.1234198 |                   1.0000000 |
| S12F-CANDIDATE-02 | A_TO_B      | BETA_VS_PRIOR   | BETA_STRUCTURE           | TRAINING_PRIOR       |           -0.0083228 |                  -0.0426739 |                   0.0258266 |                   0.3242188 |
| S12F-CANDIDATE-02 | A_TO_B      | FULL_VS_PRIOR   | FULL_STATE_GRAPH_HISTORY | TRAINING_PRIOR       |            0.1222696 |                   0.0839362 |                   0.1672970 |                   1.0000000 |
| S12F-CANDIDATE-02 | A_TO_B      | FULL_VS_DIRECT  | FULL_STATE_GRAPH_HISTORY | DIRECT_HISTORY_PHASE |            0.0368539 |                   0.0219870 |                   0.0524182 |                   1.0000000 |
| S12F-CANDIDATE-02 | B_TO_A      | DIRECT_VS_PRIOR | DIRECT_HISTORY_PHASE     | TRAINING_PRIOR       |            0.0802926 |                   0.0411416 |                   0.1210076 |                   1.0000000 |
| S12F-CANDIDATE-02 | B_TO_A      | BETA_VS_PRIOR   | BETA_STRUCTURE           | TRAINING_PRIOR       |           -0.0095494 |                  -0.0482375 |                   0.0261477 |                   0.3024902 |
| S12F-CANDIDATE-02 | B_TO_A      | FULL_VS_PRIOR   | FULL_STATE_GRAPH_HISTORY | TRAINING_PRIOR       |            0.1148889 |                   0.0763529 |                   0.1583611 |                   1.0000000 |
| S12F-CANDIDATE-02 | B_TO_A      | FULL_VS_DIRECT  | FULL_STATE_GRAPH_HISTORY | DIRECT_HISTORY_PHASE |            0.0345964 |                   0.0208846 |                   0.0493550 |                   1.0000000 |
| S12F-CANDIDATE-03 | A_TO_B      | DIRECT_VS_PRIOR | DIRECT_HISTORY_PHASE     | TRAINING_PRIOR       |            0.0907543 |                   0.0569086 |                   0.1287707 |                   1.0000000 |
| S12F-CANDIDATE-03 | A_TO_B      | BETA_VS_PRIOR   | BETA_STRUCTURE           | TRAINING_PRIOR       |           -0.0229675 |                  -0.0588531 |                   0.0104517 |                   0.0905762 |
| S12F-CANDIDATE-03 | A_TO_B      | FULL_VS_PRIOR   | FULL_STATE_GRAPH_HISTORY | TRAINING_PRIOR       |            0.1320573 |                   0.0907787 |                   0.1776161 |                   1.0000000 |
| S12F-CANDIDATE-03 | A_TO_B      | FULL_VS_DIRECT  | FULL_STATE_GRAPH_HISTORY | DIRECT_HISTORY_PHASE |            0.0413030 |                   0.0246504 |                   0.0579043 |                   1.0000000 |
| S12F-CANDIDATE-03 | B_TO_A      | DIRECT_VS_PRIOR | DIRECT_HISTORY_PHASE     | TRAINING_PRIOR       |            0.0857899 |                   0.0491946 |                   0.1250058 |                   1.0000000 |
| S12F-CANDIDATE-03 | B_TO_A      | BETA_VS_PRIOR   | BETA_STRUCTURE           | TRAINING_PRIOR       |           -0.0282386 |                  -0.0644864 |                   0.0043312 |                   0.0468750 |
| S12F-CANDIDATE-03 | B_TO_A      | FULL_VS_PRIOR   | FULL_STATE_GRAPH_HISTORY | TRAINING_PRIOR       |            0.1304718 |                   0.0879825 |                   0.1760380 |                   1.0000000 |
| S12F-CANDIDATE-03 | B_TO_A      | FULL_VS_DIRECT  | FULL_STATE_GRAPH_HISTORY | DIRECT_HISTORY_PHASE |            0.0446820 |                   0.0294851 |                   0.0593737 |                   1.0000000 |

## Overall and within-matrix ranks

| candidateId       | direction   | modelId                  |   qSpearman |   qSpearmanLower95 |   qSpearmanUpper95 |   centeredQSpearman |   centeredQSpearmanLower95 |   centeredQSpearmanUpper95 |
|:------------------|:------------|:-------------------------|------------:|-------------------:|-------------------:|--------------------:|---------------------------:|---------------------------:|
| S12F-CANDIDATE-02 | A_TO_B      | TRAINING_PRIOR           | nan         |        nan         |        nan         |         nan         |                nan         |                nan         |
| S12F-CANDIDATE-02 | A_TO_B      | DIRECT_HISTORY_PHASE     |   0.7353013 |          0.6182566 |          0.8170327 |           0.4183021 |                  0.2759033 |                  0.5446465 |
| S12F-CANDIDATE-02 | A_TO_B      | BETA_STRUCTURE           |   0.2550986 |          0.0173887 |          0.4559040 |           0.0259431 |                 -0.0201804 |                  0.0860373 |
| S12F-CANDIDATE-02 | A_TO_B      | FULL_STATE_GRAPH_HISTORY |   0.8554559 |          0.7766135 |          0.9099027 |           0.6881931 |                  0.5792224 |                  0.7680808 |
| S12F-CANDIDATE-02 | B_TO_A      | TRAINING_PRIOR           | nan         |        nan         |        nan         |         nan         |                nan         |                nan         |
| S12F-CANDIDATE-02 | B_TO_A      | DIRECT_HISTORY_PHASE     |   0.7335984 |          0.6172069 |          0.8137146 |           0.3782808 |                  0.2124427 |                  0.5222143 |
| S12F-CANDIDATE-02 | B_TO_A      | BETA_STRUCTURE           |   0.2619887 |          0.0321973 |          0.4539089 |           0.0273636 |                 -0.0165210 |                  0.0684317 |
| S12F-CANDIDATE-02 | B_TO_A      | FULL_STATE_GRAPH_HISTORY |   0.8474784 |          0.7685463 |          0.9000818 |           0.6650280 |                  0.5554925 |                  0.7493115 |
| S12F-CANDIDATE-03 | A_TO_B      | TRAINING_PRIOR           | nan         |        nan         |        nan         |         nan         |                nan         |                nan         |
| S12F-CANDIDATE-03 | A_TO_B      | DIRECT_HISTORY_PHASE     |   0.7269512 |          0.5976990 |          0.8160222 |           0.4575992 |                  0.3100542 |                  0.5868614 |
| S12F-CANDIDATE-03 | A_TO_B      | BETA_STRUCTURE           |   0.0556491 |         -0.2138492 |          0.3081605 |           0.0041556 |                 -0.0241544 |                  0.0384246 |
| S12F-CANDIDATE-03 | A_TO_B      | FULL_STATE_GRAPH_HISTORY |   0.8746284 |          0.8027720 |          0.9193227 |           0.7154814 |                  0.6021075 |                  0.7949578 |
| S12F-CANDIDATE-03 | B_TO_A      | TRAINING_PRIOR           | nan         |        nan         |        nan         |         nan         |                nan         |                nan         |
| S12F-CANDIDATE-03 | B_TO_A      | DIRECT_HISTORY_PHASE     |   0.6984221 |          0.5595446 |          0.7899063 |           0.3897785 |                  0.2405196 |                  0.5235057 |
| S12F-CANDIDATE-03 | B_TO_A      | BETA_STRUCTURE           |  -0.0153962 |         -0.2741198 |          0.2355534 |           0.0084089 |                 -0.0259363 |                  0.0424830 |
| S12F-CANDIDATE-03 | B_TO_A      | FULL_STATE_GRAPH_HISTORY |   0.8625301 |          0.7885070 |          0.9125486 |           0.7091934 |                  0.5954791 |                  0.7932418 |

## Whole-matrix permutation controls

| candidateId       | direction   | modelId                  |   observedQSpearman |   observedCenteredQSpearman |   overallUpperTailP |   centeredUpperTailP |   permutations | wholeMatrixTrajectoryPermutation   |
|:------------------|:------------|:-------------------------|--------------------:|----------------------------:|--------------------:|---------------------:|---------------:|:-----------------------------------|
| S12F-CANDIDATE-02 | A_TO_B      | BETA_STRUCTURE           |           0.2550986 |                   0.0259431 |           0.0311891 |            0.0506823 |            512 | True                               |
| S12F-CANDIDATE-02 | A_TO_B      | DIRECT_HISTORY_PHASE     |           0.7353013 |                   0.4183021 |           0.0019493 |            0.0019493 |            512 | True                               |
| S12F-CANDIDATE-02 | A_TO_B      | FULL_STATE_GRAPH_HISTORY |           0.8554559 |                   0.6881931 |           0.0019493 |            0.0019493 |            512 | True                               |
| S12F-CANDIDATE-02 | B_TO_A      | BETA_STRUCTURE           |           0.2619887 |                   0.0273636 |           0.0272904 |            0.0526316 |            512 | True                               |
| S12F-CANDIDATE-02 | B_TO_A      | DIRECT_HISTORY_PHASE     |           0.7335984 |                   0.3782808 |           0.0019493 |            0.0019493 |            512 | True                               |
| S12F-CANDIDATE-02 | B_TO_A      | FULL_STATE_GRAPH_HISTORY |           0.8474784 |                   0.6650280 |           0.0019493 |            0.0019493 |            512 | True                               |
| S12F-CANDIDATE-03 | A_TO_B      | BETA_STRUCTURE           |           0.0556491 |                   0.0041556 |           0.3606238 |            0.4346979 |            512 | True                               |
| S12F-CANDIDATE-03 | A_TO_B      | DIRECT_HISTORY_PHASE     |           0.7269512 |                   0.4575992 |           0.0019493 |            0.0019493 |            512 | True                               |
| S12F-CANDIDATE-03 | A_TO_B      | FULL_STATE_GRAPH_HISTORY |           0.8746284 |                   0.7154814 |           0.0019493 |            0.0019493 |            512 | True                               |
| S12F-CANDIDATE-03 | B_TO_A      | BETA_STRUCTURE           |          -0.0153962 |                   0.0084089 |           0.5711501 |            0.3606238 |            512 | True                               |
| S12F-CANDIDATE-03 | B_TO_A      | DIRECT_HISTORY_PHASE     |           0.6984221 |                   0.3897785 |           0.0019493 |            0.0019493 |            512 | True                               |
| S12F-CANDIDATE-03 | B_TO_A      | FULL_STATE_GRAPH_HISTORY |           0.8625301 |                   0.7091934 |           0.0019493 |            0.0019493 |            512 | True                               |

## Scientific gates

| gateId                                | candidateId       | gateFamily                |   minimumProperScoreLower95 |   minimumQSpearmanLower95 |   minimumCenteredQSpearmanLower95 |   maximumOverallPermutationP |   maximumCenteredPermutationP | passed   |
|:--------------------------------------|:------------------|:--------------------------|----------------------------:|--------------------------:|----------------------------------:|-----------------------------:|------------------------------:|:---------|
| STABLE_CAPACITY::S12F-CANDIDATE-02    | S12F-CANDIDATE-02 | STABLE_CATALYTIC_CAPACITY |                  -0.0482375 |                 0.0173887 |                       nan         |                    0.0311891 |                   nan         | False    |
| STATE_LOCAL_PROXY::S12F-CANDIDATE-02  | S12F-CANDIDATE-02 | STATE_LOCAL_PAST_PROXY    |                   0.0208846 |                 0.7685463 |                         0.5554925 |                    0.0019493 |                     0.0019493 | True     |
| STABLE_CAPACITY::S12F-CANDIDATE-03    | S12F-CANDIDATE-03 | STABLE_CATALYTIC_CAPACITY |                  -0.0644864 |                -0.2741198 |                       nan         |                    0.5711501 |                   nan         | False    |
| STATE_LOCAL_PROXY::S12F-CANDIDATE-03  | S12F-CANDIDATE-03 | STATE_LOCAL_PAST_PROXY    |                   0.0246504 |                 0.7885070 |                         0.5954791 |                    0.0019493 |                     0.0019493 | True     |
| COMPLETE_CROSS_CANDIDATE_ADJUDICATION | BOTH              | COMPLETE                  |                 nan         |               nan         |                       nan         |                  nan         |                   nan         | True     |

## Highest-weight full-state coordinates

| candidateId       | featureName                                      |   absoluteCoefficient |
|:------------------|:-------------------------------------------------|----------------------:|
| S12F-CANDIDATE-02 | latestParentDaughterH                            |             0.9945263 |
| S12F-CANDIDATE-02 | prefixInheritanceFraction                        |             0.7044153 |
| S12F-CANDIDATE-02 | normalizedGeneration                             |             0.4685220 |
| S12F-CANDIDATE-02 | currentInheritanceState                          |             0.3870091 |
| S12F-CANDIDATE-02 | recentFiveInheritanceFraction                    |             0.2559134 |
| S12F-CANDIDATE-02 | correlation__log_boost__join_share               |             0.0659615 |
| S12F-CANDIDATE-02 | completed_fissions_fraction                      |             0.0597382 |
| S12F-CANDIDATE-02 | landmark_fraction                                |             0.0597382 |
| S12F-CANDIDATE-02 | batch_step_fraction                              |             0.0553805 |
| S12F-CANDIDATE-02 | loss_share__q100                                 |             0.0549964 |
| S12F-CANDIDATE-03 | latestParentDaughterH                            |             3.3049376 |
| S12F-CANDIDATE-03 | normalizedGeneration                             |             1.1853935 |
| S12F-CANDIDATE-03 | prefixInheritanceFraction                        |             1.1635461 |
| S12F-CANDIDATE-03 | currentInheritanceState                          |             0.3514315 |
| S12F-CANDIDATE-03 | recentFiveInheritanceFraction                    |             0.1341384 |
| S12F-CANDIDATE-03 | landmark_fraction                                |             0.0907351 |
| S12F-CANDIDATE-03 | completed_fissions_fraction                      |             0.0907351 |
| S12F-CANDIDATE-03 | batch_step_fraction                              |             0.0874491 |
| S12F-CANDIDATE-03 | join_share__q100                                 |             0.0511907 |
| S12F-CANDIDATE-03 | message__RAW_FROBENIUS__COMPOSITION__k4__entropy |             0.0499501 |

## Interpretation boundary

An overall beta-only signal means catalytic matrices differ stably in their propensity to break and re-establish heredity; it is not a trajectory-local rise toward a replicator. A full-state result must additionally preserve within-matrix ordering. This adaptive reused-cohort analysis is not confirmation. A null constrains only the fixed direct-history and L34 graph summaries; it does not make the empirical committor unreal or prove that every possible observable is uninformative. Branch-derived L52 models remain forward-simulation ceilings and never enter a past-observable predictor.

## Runtime and provenance

- Repository lock: `a2208cdf1aaea14220e7ba50295a81ff114ea592`.
- Workers: `8` with one numerical-library thread; GPU hours: 0.
- Wall time: `178.937` minutes; CPU upper estimate: `23.858315` hours.
- Frozen matrices/states: `80` / `800`.
- New matrices, trajectories and branch streams: 0, 0 and 0.
- Matrix bootstraps: 4096; whole-matrix permutations: 512; exact analysis passes: 2.

## Limitations

The feature students are adaptive follow-ups to L52 and use the same L50 matrix cohort. There are only 40 development and 40 validation matrices, with five correlated states per matrix. PCA and model fitting are development-only, but the representation family was motivated by earlier E01 failures. Beta-only predictions cannot order states within a matrix. The full-state graph is permutation invariant and compact, so species-specific higher-order structure may remain compressed. The stochastic future remains represented only through the response counts.
