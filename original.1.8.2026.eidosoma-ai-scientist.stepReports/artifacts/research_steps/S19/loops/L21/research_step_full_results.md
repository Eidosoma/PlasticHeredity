# S19-L21 — Discrete-Time Survival Reconstruction of Recurring-Attractor Onset

## Chief/human handoff

- **Step:** `E01-S19-L21-DISCRETE-TIME-ATTRACTOR-ONSET-SURVIVAL-v1.0.0`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** `ATTRACTOR_ONSET_SURVIVAL_TASK_ESTABLISHED`, `SURVIVAL_REFORMULATION_NON_SUPPORT`, `SINGLE_HORIZON_INFORMATION_LOSS_NOT_PRIMARY`, `NOT_PROMOTABLE_AS_CONFIRMED`, `POSSIBLE_STABILITY_PROXY`
- **Selected discovery lead:** `NONE`.
- **Validation:** exact frozen target/feature/split replay, risk-set fixtures, matrix-grouped repeated CV, 4,096 bootstraps, 512 max-statistic endpoint permutations, feature permutation, suffix-integrity, exact model regeneration, immutable-prior, storage and artifact hashes passed.
- **Recommended next bounded loop:** Advance to one fixed outcome-blind representation loop in L22; the survival reformulation did not rescue the frozen feature families.

## Frozen question

Does modelling onset timing across four fixed post-landmark hazard intervals recover a common prefix organization signal that the single 64-to-192 binary endpoint discarded?

## Task support

| candidateId       |   atRisk |   eventsBy320 |   medianObservedTime |   censoredBy320 |
|:------------------|---------:|--------------:|---------------------:|----------------:|
| S12F-CANDIDATE-02 |       53 |            42 |                161   |              11 |
| S12F-CANDIDATE-03 |       54 |            45 |                166.5 |               9 |

The target remains the frozen completed-run recurring-attractor reconstruction. It is used only as an outcome; every predictor is fixed at observation 64 and suffix invariant.

## Methods

L21 reused the exact L18/L19/L20 prefix arrays and exact matrix splits. A fixed L2 logistic discrete-time hazard model was trained on risk-set rows for intervals ending at 128, 192, 256 and 320. Hazard products yielded cumulative risk. Primary uncertainty used catalytic-matrix bootstrap and whole-endpoint permutations; molecular observations and interval rows were never treated as independent scientific units.

## Results

| candidateId       | modelId                  |   CINDEX |   INTEGRATED_BRIER |   AUROC_128 |   AUROC_192 |   AUROC_256 |   AUROC_320 |
|:------------------|:-------------------------|---------:|-------------------:|------------:|------------:|------------:|------------:|
| S12F-CANDIDATE-02 | COMPACT_BASELINE         | 0.494314 |           0.238527 |    0.490476 |    0.468182 |    0.447635 |   0.439394  |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L19_ALL     | 0.398029 |           0.339782 |    0.4      |    0.322727 |    0.336149 |   0.30303   |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L19_L20_ALL | 0.450341 |           0.308225 |    0.393651 |    0.410606 |    0.430743 |   0.443723  |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_ALL     | 0.49583  |           0.278115 |    0.404762 |    0.477273 |    0.503378 |   0.5671    |
| S12F-CANDIDATE-02 | DUMMY_BASE_HAZARD        | 0.37301  |           0.211834 |    0.154762 |    0.45     |    0.290541 |   0.0995671 |
| S12F-CANDIDATE-02 | EXACT_H_STABILITY        | 0.46702  |           0.262955 |    0.498413 |    0.384848 |    0.364865 |   0.452381  |
| S12F-CANDIDATE-02 | TIME_ONLY                | 0.374526 |           0.211708 |    0.161905 |    0.45     |    0.283784 |   0.0995671 |
| S12F-CANDIDATE-03 | COMPACT_BASELINE         | 0.602019 |           0.202294 |    0.558824 |    0.63925  |    0.661734 |   0.671605  |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L19_ALL     | 0.566691 |           0.252425 |    0.544118 |    0.584416 |    0.604651 |   0.646914  |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L19_L20_ALL | 0.541456 |           0.270763 |    0.494118 |    0.582973 |    0.522199 |   0.580247  |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_ALL     | 0.59553  |           0.235144 |    0.588235 |    0.659452 |    0.553911 |   0.607407  |
| S12F-CANDIDATE-03 | DUMMY_BASE_HAZARD        | 0.396539 |           0.19649  |    0.167647 |    0.383838 |    0.103594 |   0.0765432 |
| S12F-CANDIDATE-03 | EXACT_H_STABILITY        | 0.653929 |           0.187619 |    0.648529 |    0.694084 |    0.809725 |   0.775309  |
| S12F-CANDIDATE-03 | TIME_ONLY                | 0.395818 |           0.19632  |    0.166176 |    0.388167 |    0.105708 |   0.0765432 |

## Gate adjudication

| candidateId       | modelId                  |   atRiskMatrices |   eventsBy320 |   censoredBy320 | taskEstablished   |   cIndex |   cIndexBootstrapLower95 |   integratedBrier |   deltaCIndexOverCompact |   deltaCIndexOverExactH |   integratedBrierGainOverCompact |   integratedBrierGainOverTime |   horizonAuRocImprovementCount |   familywisePermutationP |   featurePermutationCIndex | suffixInvariancePassed   | candidateDiscoveryGatePassed   |
|:------------------|:-------------------------|-----------------:|--------------:|----------------:|:------------------|---------:|-------------------------:|------------------:|-------------------------:|------------------------:|---------------------------------:|------------------------------:|-------------------------------:|-------------------------:|---------------------------:|:-------------------------|:-------------------------------|
| S12F-CANDIDATE-02 | COMPACT_PLUS_L19_ALL     |               53 |            42 |              11 | True              | 0.398029 |                 0.306011 |          0.339782 |              -0.0962851  |              -0.0689917 |                       -0.101255  |                    -0.128075  |                              0 |                 0.992203 |                   0.58605  | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_ALL     |               53 |            42 |              11 | True              | 0.49583  |                 0.399846 |          0.278115 |               0.0015163  |               0.0288097 |                       -0.039588  |                    -0.0664075 |                              3 |                 0.701754 |                   0.523124 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L19_L20_ALL |               53 |            42 |              11 | True              | 0.450341 |                 0.361653 |          0.308225 |              -0.0439727  |              -0.0166793 |                       -0.0696975 |                    -0.096517  |                              0 |                 0.916179 |                   0.589841 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L19_ALL     |               54 |            45 |               9 | False             | 0.566691 |                 0.472795 |          0.252425 |              -0.035328   |              -0.0872386 |                       -0.0501317 |                    -0.056105  |                              0 |                 0.892788 |                   0.578947 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_ALL     |               54 |            45 |               9 | False             | 0.59553  |                 0.500743 |          0.235144 |              -0.00648882 |              -0.0583994 |                       -0.0328508 |                    -0.0388241 |                              0 |                 0.769981 |                   0.526316 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L19_L20_ALL |               54 |            45 |               9 | False             | 0.541456 |                 0.43902  |          0.270763 |              -0.0605624  |              -0.112473  |                       -0.0684699 |                    -0.0744432 |                              0 |                 0.966862 |                   0.519827 | True                     | False                          |

The discovery gate required the same bundle in both candidates, concordance at least 0.65 with bootstrap lower bound above 0.5, better integrated Brier than time and compact controls, concordance improvements over compact and exact-H controls, better horizon AUROC at three of four horizons, family-wise endpoint-permutation `p<=0.10`, worse feature-permutation performance, and suffix integrity.

## Interpretation

This loop asks whether timing information, not a new feature or label, was missing. A null constrains the fixed survival formulation and frozen feature bundles. It does not rule out organization precursors learned by an outcome-blind representation or under a larger discovery cohort.

## Runtime and provenance

- Repository lock: `6a4c39c8c49a069c3994485d75b49ee1b0c0f86c`.
- CPU float64, `8` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `2059.551`; process CPU hours: `0.359624`.

## Autonomous continuation boundary

L21 is frozen. One next bounded loop may proceed under the existing authorization through L42. S20, E02, author contact, intervention and report-bundle work remain inactive.
