# S19-L19 — Source-Grounded Multivariate Early-Warning Observables

## Chief/human handoff

- **Step:** `E01-S19-L19-SOURCE-GROUNDED-EARLY-WARNING-OBSERVABLES-v1.0.0`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** `ATTRACTOR_ONSET_TASK_ESTABLISHED`, `EARLY_WARNING_FAMILY_NON_SUPPORT`, `CRITICAL_SLOWING_NOT_INCREMENTAL`, `RQA_NOT_INCREMENTAL`, `DMD_NOT_INCREMENTAL`, `NOT_PROMOTABLE_AS_CONFIRMED`, `POSSIBLE_STABILITY_PROXY`
- **Selected discovery lead:** `NONE`.
- **Validation:** exact L18 task/split replay, immutable-prior validation, independent prefix-feature replay, exact suffix invariance, matrix-level repeated CV, 4,096 bootstraps, 512 max-statistic label permutations, temporal/feature controls, regeneration, storage and artifact hashes passed.
- **Recommended next bounded loop:** Advance to a nonduplicative multiscale geometry/topology loop; classical critical-slowing, fixed-threshold RQA, and local DMD are pruned on this task.

## Frozen question

Do classical critical-slowing indicators, recurrence-plot line topology, or local linear/DMD relaxation diagnostics calculated only from observations 0–63 predict first entry into the frozen recurring-attractor state during observations 64–191, beyond time, exact adjacent H/stability and prefix recurrence geometry?

## Cohort

| candidateId       |   atRisk |   events |   occupancy |   nonEvents |
|:------------------|---------:|---------:|------------:|------------:|
| S12F-CANDIDATE-02 |       53 |       33 |    0.207904 |          20 |
| S12F-CANDIDATE-03 |       54 |       33 |    0.213186 |          21 |

This is the exact L18 task. The lower-occupancy outcome creates real event/non-event support, but its completed-run attractor definition remains retrospective and author-ambiguous.

## Methods

L19 froze three published method families before outcomes: (1) lag-one autocorrelation, variance, covariance-spectrum concentration and spectral reddening; (2) recurrence rate, determinism, entropy, laminarity and trapping time at the unchanged `H=0.9`; and (3) rank-at-most-eight ridge-stabilized DMD reconstruction, spectral-radius, effective-rank and nonnormality diagnostics. Full-prefix values and a fixed last-32-minus-first-32 contrast were used. The estimator remained an untuned `C=1` L2 logistic regression with training-only imputation and scaling on the exact L18 splits.

## Results

| candidateId       | modelId                    |    AUROC |    AUPRC |    BRIER |   BALANCED_ACCURACY |
|:------------------|:---------------------------|---------:|---------:|---------:|--------------------:|
| S12F-CANDIDATE-02 | COMPACT_BASELINE           | 0.419697 | 0.619252 | 0.277271 |            0.509091 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_ALL           | 0.339394 | 0.534874 | 0.372716 |            0.422727 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_DMD           | 0.433333 | 0.600731 | 0.319742 |            0.513636 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_EWS           | 0.4      | 0.579795 | 0.322743 |            0.37803  |
| S12F-CANDIDATE-02 | COMPACT_PLUS_RQA           | 0.392424 | 0.565412 | 0.29791  |            0.438636 |
| S12F-CANDIDATE-02 | DUMMY_TRAINING_PRIOR       | 0.426515 | 0.600295 | 0.235112 |            0.5      |
| S12F-CANDIDATE-02 | EXACT_H_STABILITY          | 0.371212 | 0.587305 | 0.305521 |            0.438636 |
| S12F-CANDIDATE-02 | PREFIX_RECURRENCE_GEOMETRY | 0.431818 | 0.62959  | 0.27831  |            0.403788 |
| S12F-CANDIDATE-03 | COMPACT_BASELINE           | 0.619048 | 0.746198 | 0.237519 |            0.608225 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_ALL           | 0.520924 | 0.673222 | 0.310456 |            0.484848 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_DMD           | 0.580087 | 0.694497 | 0.267953 |            0.595238 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_EWS           | 0.588745 | 0.730526 | 0.272538 |            0.577922 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_RQA           | 0.607504 | 0.714337 | 0.253123 |            0.562771 |
| S12F-CANDIDATE-03 | DUMMY_TRAINING_PRIOR       | 0.378066 | 0.553953 | 0.238325 |            0.5      |
| S12F-CANDIDATE-03 | EXACT_H_STABILITY          | 0.68254  | 0.792916 | 0.228637 |            0.577922 |
| S12F-CANDIDATE-03 | PREFIX_RECURRENCE_GEOMETRY | 0.568543 | 0.716961 | 0.248779 |            0.515152 |

## Gate adjudication

| candidateId       | modelId          |   atRiskMatrices |   events |   nonEvents | taskEstablished   |    auRoc |   auRocBootstrapLower95 |    auPrc |   prevalence |    brier |   dummyBrier |   deltaOverCompact |   deltaOverExactH |   familywisePermutationP |   leaveOneOutPositiveFraction |   temporalPermutationAuRoc | suffixInvariancePassed   | candidateDiscoveryGatePassed   |
|:------------------|:-----------------|-----------------:|---------:|------------:|:------------------|---------:|------------------------:|---------:|-------------:|---------:|-------------:|-------------------:|------------------:|-------------------------:|------------------------------:|---------------------------:|:-------------------------|:-------------------------------|
| S12F-CANDIDATE-02 | COMPACT_PLUS_EWS |               53 |       33 |          20 | True              | 0.4      |                0.241822 | 0.579795 |     0.622642 | 0.322743 |     0.235112 |         -0.019697  |         0.0287879 |                 0.896686 |                     0.0188679 |                   0.327273 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_RQA |               53 |       33 |          20 | True              | 0.392424 |                0.23913  | 0.565412 |     0.622642 | 0.29791  |     0.235112 |         -0.0272727 |         0.0212121 |                 0.916179 |                     0.0377358 |                   0.504545 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_DMD |               53 |       33 |          20 | True              | 0.433333 |                0.272247 | 0.600731 |     0.622642 | 0.319742 |     0.235112 |          0.0136364 |         0.0621212 |                 0.781676 |                     0.924528  |                   0.354545 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_ALL |               53 |       33 |          20 | True              | 0.339394 |                0.188854 | 0.534874 |     0.622642 | 0.372716 |     0.235112 |         -0.080303  |        -0.0318182 |                 0.990253 |                     0         |                   0.486364 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_EWS |               54 |       33 |          21 | True              | 0.588745 |                0.429632 | 0.730526 |     0.611111 | 0.272538 |     0.238325 |         -0.030303  |        -0.0937951 |                 0.931774 |                     0         |                   0.593074 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_RQA |               54 |       33 |          21 | True              | 0.607504 |                0.452365 | 0.714337 |     0.611111 | 0.253123 |     0.238325 |         -0.011544  |        -0.0750361 |                 0.846004 |                     0.037037  |                   0.584416 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_DMD |               54 |       33 |          21 | True              | 0.580087 |                0.417991 | 0.694497 |     0.611111 | 0.267953 |     0.238325 |         -0.038961  |        -0.102453  |                 0.951267 |                     0         |                   0.597403 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_ALL |               54 |       33 |          21 | True              | 0.520924 |                0.361934 | 0.673222 |     0.611111 | 0.310456 |     0.238325 |         -0.0981241 |        -0.161616  |                 0.996101 |                     0         |                   0.506494 | True                     | False                          |

The discovery gate required the same frozen model in both candidates, AUROC at least 0.65 with a bootstrap lower bound above 0.5, AUPRC above prevalence, no Brier loss against the dummy, positive increments over both the compact and exact-H baselines, family-wise max-statistic permutation `p<=0.10`, at least 90% positive leave-one-matrix-out increments, a worse temporal-permutation control, and exact suffix invariance. This is a discovery threshold, not confirmation.

## Interpretation

Published early-warning observables are plausible when an approaching transition exhibits critical slowing, changing recurrence topology, or local relaxation toward a lower-dimensional attractor. Failure here constrains those fixed implementations on this particular 64-to-192 GARD onset task; it does not prove that organization has no precursor. A one-candidate or stability-explained pattern is retained but cannot count as a solution.

No completed trajectory, target centroid, suffix statistic, molecular-row pseudoreplication, favorable-candidate pooling, or outcome-guided feature setting entered a prospective input. The target itself remains retrospectively adjudicated.

## Runtime and provenance

- Repository lock: `fcb49f4ceee34b95146c452631ad5291a7885f5b`.
- CPU float64, `8` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `1908.892`; process CPU hours: `0.418746`.
- Published source identities and reconstruction choices are in `source_grounding_registry.csv` and `source_grounding_report.md`.

## Autonomous continuation boundary

L19 is frozen. The prior human authorization permits the next single bounded, nonduplicative loop without an intermediate Chief handoff, through at most L42. No S20, E02, author contact, intervention, or report-bundle work is active.
