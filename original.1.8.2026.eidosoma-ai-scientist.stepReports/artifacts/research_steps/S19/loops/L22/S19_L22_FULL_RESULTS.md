# S19-L22 — Outcome-Blind Permutation-Invariant Prefix Representation

## Chief/human handoff

- **Step:** `E01-S19-L22-OUTCOME-BLIND-PREFIX-REPRESENTATION-v1.0.0`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** `ATTRACTOR_ONSET_TASK_ESTABLISHED`, `OUTCOME_BLIND_REPRESENTATION_NON_SUPPORT`, `RANDOM_CONVOLUTION_NOT_INCREMENTAL`, `NOT_PROMOTABLE_AS_CONFIRMED`, `POSSIBLE_STABILITY_PROXY`
- **Selected discovery lead:** `NONE`.
- **Validation:** one pre-outcome kernel bank, exact target/split replay, immutable-prior validation through L21, mandatory fixtures, independent all-unit representation replay, molecule-label invariance, suffix invariance, matrix repeated CV, 4,096 bootstraps, 512 whole-matrix permutations, temporal/feature controls, regeneration, storage and artifact hashes passed.
- **Recommended next bounded loop:** Advance to a larger, independently generated discovery cohort in L23 before inventing another feature family; the current 53/54-matrix task cannot distinguish stable weak effects from candidate heterogeneity.

## Frozen question

Does one outcome-blind random-convolution representation of observations 0–63 add candidate-consistent information before first recurring-attractor entry during observations 64–191 beyond compact ordinary dynamics and exact H/stability?

## Cohort

| candidateId       |   atRisk |   events |   occupancy |   nonEvents |
|:------------------|---------:|---------:|------------:|------------:|
| S12F-CANDIDATE-02 |       53 |       33 |    0.207904 |          20 |
| S12F-CANDIDATE-03 |       54 |       33 |    0.213186 |          21 |

The target is the frozen completed-run L02 recurring-attractor reconstruction and remains retrospective and author-ambiguous. Every competitive input is prefix-only.

## Methods

Eleven permutation-invariant organization channels encode mass, diversity/concentration, adjacent motion, past recurrence and running-centroid similarity. Each trajectory's channels are standardized without cohort or outcome information. One bank of 64 fixed mean-centered unit-norm Gaussian kernels (lengths 7/9/11; dilations 1/2/4/8; frozen biases) emits maximum and proportion-positive summaries, yielding 128 features. The bank identity is `3de958f6be47bb563b30ea07e0099dcd4642c1ebcebb050ae688884f606f45c1`. The unchanged C=1 L2 logistic model and exact L18 splits were used. No bank, channel, bias, kernel count or downstream hyperparameter was selected from outcomes.

## Results

| candidateId       | modelId                         |    AUROC |    AUPRC |    BRIER |   BALANCED_ACCURACY |
|:------------------|:--------------------------------|---------:|---------:|---------:|--------------------:|
| S12F-CANDIDATE-02 | COMPACT_BASELINE                | 0.419697 | 0.619252 | 0.277271 |            0.509091 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_RANDOM_CONVOLUTION | 0.54697  | 0.696592 | 0.2896   |            0.503788 |
| S12F-CANDIDATE-02 | DUMMY_TRAINING_PRIOR            | 0.426515 | 0.600295 | 0.235112 |            0.5      |
| S12F-CANDIDATE-02 | EXACT_H_STABILITY               | 0.371212 | 0.587305 | 0.305521 |            0.438636 |
| S12F-CANDIDATE-02 | RANDOM_CONVOLUTION_ONLY         | 0.548485 | 0.704409 | 0.285357 |            0.493939 |
| S12F-CANDIDATE-03 | COMPACT_BASELINE                | 0.619048 | 0.746198 | 0.237519 |            0.608225 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_RANDOM_CONVOLUTION | 0.532468 | 0.662088 | 0.299099 |            0.556277 |
| S12F-CANDIDATE-03 | DUMMY_TRAINING_PRIOR            | 0.378066 | 0.553953 | 0.238325 |            0.5      |
| S12F-CANDIDATE-03 | EXACT_H_STABILITY               | 0.68254  | 0.792916 | 0.228637 |            0.577922 |
| S12F-CANDIDATE-03 | RANDOM_CONVOLUTION_ONLY         | 0.481962 | 0.63421  | 0.335885 |            0.532468 |

## Gate adjudication

| candidateId       | modelId                         |   atRiskMatrices |   events |   nonEvents | taskEstablished   |    auRoc |   auRocBootstrapLower95 |    auPrc |   prevalence |    brier |   dummyBrier |   deltaOverCompact |   deltaOverExactH |   familywisePermutationP |   leaveOneOutPositiveFraction |   temporalPermutationAuRoc | suffixInvariancePassed   | candidateDiscoveryGatePassed   |
|:------------------|:--------------------------------|-----------------:|---------:|------------:|:------------------|---------:|------------------------:|---------:|-------------:|---------:|-------------:|-------------------:|------------------:|-------------------------:|------------------------------:|---------------------------:|:-------------------------|:-------------------------------|
| S12F-CANDIDATE-02 | COMPACT_PLUS_RANDOM_CONVOLUTION |               53 |       33 |          20 | True              | 0.54697  |                0.377548 | 0.696592 |     0.622642 | 0.2896   |     0.235112 |          0.127273  |          0.175758 |                 0.19883  |                             1 |                   0.398485 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_RANDOM_CONVOLUTION |               54 |       33 |          21 | True              | 0.532468 |                0.372339 | 0.662088 |     0.611111 | 0.299099 |     0.238325 |         -0.0865801 |         -0.150072 |                 0.793372 |                             0 |                   0.562771 | True                     | False                          |

The discovery gate required the same frozen model to pass in both candidates, AUROC at least 0.65 with bootstrap lower bound above 0.5, AUPRC above prevalence, no Brier loss, positive increments over compact and exact-H controls, matrix-permutation p<=0.10, at least 90% positive leave-one-out increments, worse temporal-permutation performance, and exact suffix invariance.

## Interpretation

This is an outcome-blind nonlinear map of past organization motifs, not a fitted causal-emergence measure. A null constrains this one fixed representation on the studied cohort; it does not prove that no pre-onset organization exists. A favorable studied-cohort result is not a solution until untouched confirmation passes.

## Runtime and provenance

- Repository lock: `c29b2144ade7b6d47b24cc6544755f199fb09f8f`.
- CPU float64, `8` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `1031.041`; process CPU hours: `0.249765`.

## Autonomous continuation boundary

L22 is frozen. The existing authorization permits one next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
