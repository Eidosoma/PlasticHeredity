# S19-L35 — Short-Branch Ensemble Mechanism Attribution

## Chief/human handoff

- **Step:** `E01-S19-L35-SHORT-BRANCH-ENSEMBLE-MECHANISM-ATTRIBUTION-v1.0.0`
- **Status:** complete under the authorized L19–L42 sequence.
- **Classifications:** `NO_UNIVERSAL_SHORT_BRANCH_MECHANISM`, `SHOOTING_TEACHER_NOT_DISTILLED_TO_PREENTRY_MECHANISM`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** exact replay of 17,920 frozen H8 streams and 143,360 generated observations; exact compact path, entry, terminal and stream identities; exact 280 state/beta/clock/target/q inputs; deterministic full regeneration; independent branch halves; 4,096 catalytic-matrix bootstraps; immutable, runtime, storage and artifact hashes.
- **Recommended next action:** `TARGET_BASIN_TRANSFER_INDEPENDENT_REFERENCE_AUDIT`.

## Frozen question

What information appears during the successful eight-observation stochastic propagation that the tested present-state and observed-prefix summaries discarded? L35 does not create another predictor. It replays the already established L30/L31 teacher branches and records every intermediate physical update. Cumulative entry and similarity to the completed-run basin are explicitly separated from five branch-physical summaries. A future-branch measurement is not a past-observable biomarker.

## Inputs and method

- 200 L28 states (development and validation) and 80 untouched L31 confirmation states.
- Exactly 64 existing H8 continuations per state, split prospectively into two 32-branch halves.
- Exactly eight selected-clock observations per continuation; zero new branch streams, matrices, trajectories, targets, thresholds or simulator settings.
- Candidate 2 and candidate 3 and all three cohorts remain separate.
- Nine registered state-offset metrics; the physical solution gate uses exactly five metrics, the same metric and offset no later than offset four, and all four candidate/evaluation-cohort groups.
- H32 empirical committor is the response; catalytic matrix/state is the independent higher-level unit.

## Main result

The common physical-mechanism gate was `FAIL`. The strongest registered row in each evaluation group was:

| evaluationCohort   | candidateId       |   offset | metricId                   |   spearmanH32 |   spearmanLower95 |
|:-------------------|:------------------|---------:|:---------------------------|--------------:|------------------:|
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        8 | cumulativeEntryFraction    |      0.848893 |          0.705116 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        7 | cumulativeEntryFraction    |      0.835027 |          0.683852 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        6 | cumulativeEntryFraction    |      0.828677 |          0.676778 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        5 | cumulativeEntryFraction    |      0.792222 |          0.632246 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        8 | cumulativeEntryFraction    |      0.766359 |          0.571412 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        8 | cumulativeEntryFraction    |      0.754172 |          0.595716 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        7 | cumulativeEntryFraction    |      0.751646 |          0.595927 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        6 | cumulativeEntryFraction    |      0.750724 |          0.591576 |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        8 | cumulativeEntryFraction    |      0.743224 |          0.573718 |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        7 | cumulativeEntryFraction    |      0.737813 |          0.559412 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        4 | cumulativeEntryFraction    |      0.737449 |          0.548133 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        7 | cumulativeEntryFraction    |      0.730484 |          0.524106 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        5 | cumulativeEntryFraction    |      0.727197 |          0.56298  |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        6 | cumulativeEntryFraction    |      0.692706 |          0.519895 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        6 | cumulativeEntryFraction    |      0.67062  |          0.485001 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        7 | atRiskMeanTargetScore      |      0.664071 |          0.452881 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        4 | allBranchTargetScoreSd     |      0.659652 |          0.503518 |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        5 | cumulativeEntryFraction    |      0.639352 |          0.455377 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        6 | allBranchTargetScoreSd     |      0.60234  |          0.37522  |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        8 | atRiskMeanJoinShareMaximum |      0.474119 |          0.193354 |

The strongest *physical* row in each group was:

| evaluationCohort   | candidateId       |   offset | metricId                     |   spearmanH32 |   spearmanLower95 |
|:-------------------|:------------------|---------:|:-----------------------------|--------------:|------------------:|
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        1 | atRiskMeanGrossSampledEvents |      0.594535 |          0.330254 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        7 | atRiskMeanJoinShareMaximum   |      0.481515 |          0.141434 |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        8 | atRiskMeanJoinShareMaximum   |      0.474119 |          0.193354 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        7 | fissionFractionAtOffset      |      0.400815 |          0.138585 |

The locked common-gate matrix is:

| metricId                               | metricClass                     |   offset |   minimumSpearmanH32 |   minimumLower95 | commonGatePassed   |
|:---------------------------------------|:--------------------------------|---------:|---------------------:|-----------------:|:-------------------|
| cumulativeEntryFraction                | TEACHER_REALIZED_ENTRY          |        1 |           0.193206   |       0.153711   | False              |
| cumulativeEntryFraction                | TEACHER_REALIZED_ENTRY          |        2 |           0.22714    |      -0.00695192 | False              |
| cumulativeEntryFraction                | TEACHER_REALIZED_ENTRY          |        3 |           0.287109   |       0.0426276  | False              |
| cumulativeEntryFraction                | TEACHER_REALIZED_ENTRY          |        4 |           0.400669   |       0.166832   | False              |
| atRiskMeanTargetScore                  | RETROSPECTIVE_BASIN_CONDITIONED |        1 |           0.258279   |      -0.0265514  | False              |
| atRiskMeanTargetScore                  | RETROSPECTIVE_BASIN_CONDITIONED |        2 |           0.279954   |       0.00947054 | False              |
| atRiskMeanTargetScore                  | RETROSPECTIVE_BASIN_CONDITIONED |        3 |           0.336262   |       0.054563   | False              |
| atRiskMeanTargetScore                  | RETROSPECTIVE_BASIN_CONDITIONED |        4 |           0.351036   |       0.0585249  | False              |
| atRiskMeanTargetScoreChangeFromCurrent | RETROSPECTIVE_BASIN_CONDITIONED |        1 |           0.275849   |      -0.0364961  | False              |
| atRiskMeanTargetScoreChangeFromCurrent | RETROSPECTIVE_BASIN_CONDITIONED |        2 |           0.132054   |      -0.217578   | False              |
| atRiskMeanTargetScoreChangeFromCurrent | RETROSPECTIVE_BASIN_CONDITIONED |        3 |           0.283742   |      -0.0489927  | False              |
| atRiskMeanTargetScoreChangeFromCurrent | RETROSPECTIVE_BASIN_CONDITIONED |        4 |           0.289473   |      -0.0896413  | False              |
| allBranchTargetScoreSd                 | RETROSPECTIVE_BASIN_CONDITIONED |        1 |           0.146005   |      -0.165291   | False              |
| allBranchTargetScoreSd                 | RETROSPECTIVE_BASIN_CONDITIONED |        2 |           0.324554   |       0.0185631  | False              |
| allBranchTargetScoreSd                 | RETROSPECTIVE_BASIN_CONDITIONED |        3 |           0.270246   |      -0.0460388  | False              |
| allBranchTargetScoreSd                 | RETROSPECTIVE_BASIN_CONDITIONED |        4 |           0.305955   |       0.00714942 | False              |
| atRiskCompositionDispersion            | BRANCH_PHYSICAL_MECHANISM       |        1 |          -0.0727205  |      -0.402537   | False              |
| atRiskCompositionDispersion            | BRANCH_PHYSICAL_MECHANISM       |        2 |          -0.00394608 |      -0.358261   | False              |
| atRiskCompositionDispersion            | BRANCH_PHYSICAL_MECHANISM       |        3 |           0.0497429  |      -0.213024   | False              |
| atRiskCompositionDispersion            | BRANCH_PHYSICAL_MECHANISM       |        4 |          -0.0525139  |      -0.315409   | False              |
| fissionFractionAtOffset                | BRANCH_PHYSICAL_MECHANISM       |        1 |          -0.119316   |      -0.360691   | False              |
| fissionFractionAtOffset                | BRANCH_PHYSICAL_MECHANISM       |        2 |          -0.0145517  |      -0.316752   | False              |
| fissionFractionAtOffset                | BRANCH_PHYSICAL_MECHANISM       |        3 |           0.00977411 |      -0.286108   | False              |
| fissionFractionAtOffset                | BRANCH_PHYSICAL_MECHANISM       |        4 |          -0.147669   |      -0.409721   | False              |
| atRiskMeanMass                         | BRANCH_PHYSICAL_MECHANISM       |        1 |           0.0431105  |      -0.280168   | False              |
| atRiskMeanMass                         | BRANCH_PHYSICAL_MECHANISM       |        2 |          -0.0444081  |      -0.337118   | False              |
| atRiskMeanMass                         | BRANCH_PHYSICAL_MECHANISM       |        3 |          -0.0424334  |      -0.359609   | False              |
| atRiskMeanMass                         | BRANCH_PHYSICAL_MECHANISM       |        4 |          -0.0247841  |      -0.349404   | False              |
| atRiskMeanJoinShareMaximum             | BRANCH_PHYSICAL_MECHANISM       |        1 |           0.177643   |      -0.18319    | False              |
| atRiskMeanJoinShareMaximum             | BRANCH_PHYSICAL_MECHANISM       |        2 |           0.246748   |      -0.11929    | False              |
| atRiskMeanJoinShareMaximum             | BRANCH_PHYSICAL_MECHANISM       |        3 |           0.237467   |      -0.0676285  | False              |
| atRiskMeanJoinShareMaximum             | BRANCH_PHYSICAL_MECHANISM       |        4 |           0.269495   |      -0.0355196  | False              |
| atRiskMeanGrossSampledEvents           | BRANCH_PHYSICAL_MECHANISM       |        1 |           0.184422   |      -0.13436    | False              |
| atRiskMeanGrossSampledEvents           | BRANCH_PHYSICAL_MECHANISM       |        2 |          -0.0967371  |      -0.383864   | False              |
| atRiskMeanGrossSampledEvents           | BRANCH_PHYSICAL_MECHANISM       |        3 |           0.127011   |      -0.207509   | False              |
| atRiskMeanGrossSampledEvents           | BRANCH_PHYSICAL_MECHANISM       |        4 |           0.248344   |      -0.0803823  | False              |

## When branches separate

Entrant-versus-nonentrant branches were compared only while still at risk before each offset. The largest early contrasts are:

| evaluationCohort   | candidateId       |   offset | featureId    |   definedStates |   meanEntrantMinusNonentrant |    lower95 |   upper95 |
|:-------------------|:------------------|---------:|:-------------|----------------:|-----------------------------:|-----------:|----------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 |        1 | boostMaximum |              20 |                     6648.45  | -285.824   | 20157.7   |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        2 | boostMaximum |              20 |                     4528.22  |  -54.6844  | 13453.8   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        4 | boostMaximum |              18 |                     2538.4   |  700.466   |  5638.55  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        3 | boostMaximum |              18 |                     2082.67  |  673.4     |  3757.41  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        4 | boostMaximum |              17 |                     1190.7   |  507.489   |  2008.27  |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        4 | boostMaximum |              19 |                      841.45  | -507.032   |  2515.1   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        3 | boostMaximum |              17 |                      795.403 |  174.258   |  1673.51  |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        1 | boostSd      |              20 |                      659.026 |  -30.3819  |  2009.39  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        2 | boostMaximum |              18 |                      556.339 |  235.585   |   912.494 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        4 | boostMaximum |              14 |                      495.974 |  -30.7705  |  1194.94  |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        2 | boostSd      |              20 |                      450.052 |   -4.74595 |  1336.54  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        1 | boostMaximum |              18 |                      334.186 |   32.0982  |   766.738 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        3 | boostMaximum |              14 |                      283.599 | -111.588   |   857.686 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        4 | boostSd      |              18 |                      248.323 |   63.7993  |   563.556 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        2 | boostMaximum |              17 |                      241.202 | -194.882   |   619.196 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        1 | boostMaximum |              17 |                      230.662 |   24.8223  |   475.306 |

These contrasts describe what becomes visible *after stochastic propagation begins*. They cannot show that the same information is measurable at the starting state.

## Teacher attribution and reliability

The frozen L30 teacher's largest standardized coefficients were:

| candidateId       |   absoluteRankWithinCandidate | featureName                 |   standardizedCoefficient |
|:------------------|------------------------------:|:----------------------------|--------------------------:|
| S12F-CANDIDATE-02 |                             1 | q8JeffreysLogit             |                  1.2544   |
| S12F-CANDIDATE-02 |                             2 | meanMaximumTargetScore      |                  0.729648 |
| S12F-CANDIDATE-02 |                             3 | targetComponentFraction     |                  0.473696 |
| S12F-CANDIDATE-02 |                             4 | currentTargetScore          |                 -0.405071 |
| S12F-CANDIDATE-02 |                             5 | sdMaximumTargetScore        |                  0.394294 |
| S12F-CANDIDATE-03 |                             1 | q8JeffreysLogit             |                  1.47171  |
| S12F-CANDIDATE-03 |                             2 | meanMaximumTargetScore      |                  0.656255 |
| S12F-CANDIDATE-03 |                             3 | sdMaximumTargetScore        |                  0.636812 |
| S12F-CANDIDATE-03 |                             4 | fractionBranchesWithFission |                 -0.423558 |
| S12F-CANDIDATE-03 |                             5 | meanMolecularUpdates        |                 -0.300606 |

Independent 32/32 branch-half reliability was highest for:

| evaluationCohort   | candidateId       |   offset | metricId                |   splitHalfSpearman |
|:-------------------|:------------------|---------:|:------------------------|--------------------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 |        1 | fissionFractionAtOffset |            1        |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        1 | fissionFractionAtOffset |            1        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        1 | fissionFractionAtOffset |            1        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        1 | fissionFractionAtOffset |            1        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        1 | cumulativeEntryFraction |            1        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        1 | atRiskMeanTargetScore   |            0.999812 |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        1 | atRiskMeanTargetScore   |            0.999136 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |        2 | atRiskMeanTargetScore   |            0.998687 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |        2 | cumulativeEntryFraction |            0.998542 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        1 | atRiskMeanTargetScore   |            0.998367 |
| L28_VALIDATION     | S12F-CANDIDATE-02 |        2 | atRiskMeanTargetScore   |            0.998175 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |        2 | atRiskMeanTargetScore   |            0.998079 |

Reliability of a future-ensemble summary is distinct from past-observable transferability.

## Interpretation

NO_UNIVERSAL_SHORT_BRANCH_MECHANISM, SHOOTING_TEACHER_NOT_DISTILLED_TO_PREENTRY_MECHANISM, NOT_PROMOTABLE_AS_CONFIRMED. The audit distinguishes three layers: realized basin entry (teacher), completed-run basin-distance summaries (retrospective and target-conditioned), and physical branch evolution. No L35 result changes S18, supports the paper's PhiRL claim, establishes early warning, or licenses intervention or reactive-current analysis.

## Validation and reproducibility

- Repository lock: `ab5115fb5cbcacb37f83aeec8cde609e4ee6c4af` on `eidosoma/groups/42`.
- Workers: `8`; one numerical-library thread each; GPU hours `0`.
- Wall time: `811.82` seconds; controller CPU does not include worker CPU and is reported separately.
- The full scientific scope was independently regenerated from the same frozen states and streams and every table hash matched.
- Compact regenerated results were compared field by field with both L30 and L31 authoritative branch artifacts.

## Caveats

The target basin remains matrix-specific and reconstructed from each completed trajectory. Target-score quantities therefore contain retrospective basin information. The branches themselves are forward stochastic samples. There is one selected state per catalytic matrix in these cohorts, so within-matrix ordering remains unidentifiable. A physical branch feature that separates early would still require a new loop testing an outcome-blind present-state or observed-history proxy.

## Next boundary

L35 is frozen. The standing autonomous authorization permits only the narrowly named `TARGET_BASIN_TRANSFER_INDEPENDENT_REFERENCE_AUDIT` continuation if no solution boundary was reached. S20, E02, author contact, interventions, reactive-current claims and report generation remain inactive.
