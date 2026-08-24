# S19-L28 — Branched Empirical Committor Identifiability

## Chief/human handoff

- **Step:** `E01-S19-L28-BRANCHED-EMPIRICAL-COMMITTOR-IDENTIFIABILITY-v1.0.0`
- **Status:** complete under the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** `STATE_DEPENDENT_COMMITTOR_ESTABLISHED`, `EXISTING_REPRESENTATIONS_MISS_STATE_SIGNAL`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** exact restoration of all 200 selected simulator states; 25,600 unique domain-separated branches; independent 64/64 half estimates; full branch replay; frozen target, seed, source, suffix, runtime/storage, regeneration and artifact gates; 4,096 catalytic-matrix bootstraps per candidate passed.
- **Recommended next action:** Proceed to one exact-GARD-generator drift/diffusion feature loop only if existing frozen representations missed the established state signal; transition-tube/current work remains prohibited until a held-out committor-predictive coordinate exists.

## Frozen question

Does the L23/L25 first-entry target have a reproducible state-dependent probability of entry during the next 32 selected-clock observations, or were L18–L27 asking predictors to recover a target whose single-future realization is not committor-identifiable at that horizon?

## Design

Ten deterministic, unique-matrix states were selected from each candidate × development/validation × landmark stratum before branch outcomes (200 states total). Each state was restored at selected-clock index `landmark-1`, including its count vector, catalytic matrix, mass, growth/fission phase, generation-local step, candidate exposure/daughter/trim semantics and constant reservoir. The completed-run matrix-specific L23 target centroid was frozen as the basin and is explicitly `RETROSPECTIVE_COMPLETED_RUN_MATRIX_SPECIFIC`. Each state received 128 independent forward continuations; entry was evaluated over exactly 32 new selected-clock observations. Branches 0–63 and 64–127 formed prospectively independent reliability halves.

## Committor reliability

| candidateId       |   states |   matrices |   meanQHat |   medianQHat |   splitHalfSpearman |   intermediateStateCount |   intermediateStateFraction |   zeroQStateCount |   oneQStateCount |   originalSingleFutureEventPrevalence |   observedBetweenStateVariance |   estimatedBinomialNoiseVariance |   correctedBetweenStateVariance |
|:------------------|---------:|-----------:|-----------:|-------------:|--------------------:|-------------------------:|----------------------------:|------------------:|-----------------:|--------------------------------------:|-------------------------------:|---------------------------------:|--------------------------------:|
| S12F-CANDIDATE-02 |      100 |        100 |   0.251641 |    0.105469  |            0.926006 |                       42 |                        0.42 |                10 |                1 |                                  0.29 |                      0.0944446 |                      0.000746595 |                       0.093698  |
| S12F-CANDIDATE-03 |      100 |        100 |   0.216953 |    0.0859375 |            0.932538 |                       36 |                        0.36 |                16 |                2 |                                  0.33 |                      0.0863368 |                      0.000664654 |                       0.0856722 |

## Gate adjudication

| candidateId       |   states |   correctedBetweenStateVariance |   correctedVarianceBootstrapLower95 |   splitHalfSpearman |   splitHalfSpearmanBootstrapLower95 |   finiteSplitHalfBootstrapFraction |   intermediateStateCount | targetBranchable   | correctedVariancePassed   | splitHalfReliabilityPassed   | intermediateRegionPassed   | exactReplayPassed   | noUnregisteredSuffixLeakage   | candidateCommittorGatePassed   |
|:------------------|---------:|--------------------------------:|------------------------------------:|--------------------:|------------------------------------:|-----------------------------------:|-------------------------:|:-------------------|:--------------------------|:-----------------------------|:---------------------------|:--------------------|:------------------------------|:-------------------------------|
| S12F-CANDIDATE-02 |      100 |                       0.093698  |                           0.0638188 |            0.926006 |                            0.865088 |                                  1 |                       42 | True               | True                      | True                         | True                       | True                | True                          | True                           |
| S12F-CANDIDATE-03 |      100 |                       0.0856722 |                           0.0540544 |            0.932538 |                            0.883714 |                                  1 |                       36 | True               | True                      | True                         | True                       | True                | True                          | True                           |

## Existing frozen representation audit

| candidateId       | predictorId           |   states |   spearmanQHat |   binomialLogLossPerBranch |   binomialDeviancePerState |   brierScorePerBranch |   developmentPrior |   developmentPriorBrier |   brierImprovementOverPrior |   calibrationIntercept |   calibrationSlope |   calibrationMeanAbsoluteError |
|:------------------|:----------------------|---------:|---------------:|---------------------------:|---------------------------:|----------------------:|-------------------:|------------------------:|----------------------------:|-----------------------:|-------------------:|-------------------------------:|
| S12F-CANDIDATE-02 | EXACT_H_TRACE_ANALOG  |       50 |      0.298928  |                   0.551608 |                    67.8898 |              0.18431  |           0.251406 |                0.188434 |                  0.00412448 |             nan        |         nan        |                      0.0717708 |
| S12F-CANDIDATE-02 | OPERATOR_CHANGE       |       50 |      0.247945  |                   0.583857 |                    76.1455 |              0.197986 |           0.251406 |                0.188434 |                 -0.00955181 |              -0.746533 |           0.416935 |                      0.107823  |
| S12F-CANDIDATE-02 | ORDINARY_PATH_ANALOG  |       50 |      0.232078  |                   0.591805 |                    78.1804 |              0.190679 |           0.251406 |                0.188434 |                 -0.00224496 |              -0.748541 |           0.22373  |                      0.089625  |
| S12F-CANDIDATE-02 | RECURRENCE_MAP_ANALOG |       50 |      0.167309  |                   0.726656 |                   112.702  |              0.192044 |           0.251406 |                0.188434 |                 -0.00361024 |             nan        |         nan        |                      0.0957708 |
| S12F-CANDIDATE-03 | EXACT_H_TRACE_ANALOG  |       50 |      0.186226  |                   0.594398 |                    73.8348 |              0.172921 |           0.206094 |                0.176386 |                  0.00346484 |              -0.944838 |           0.162511 |                      0.0646667 |
| S12F-CANDIDATE-03 | OPERATOR_CHANGE       |       50 |      0.33658   |                   0.538851 |                    59.6146 |              0.179444 |           0.206094 |                0.176386 |                 -0.00305832 |             nan        |         nan        |                      0.0870054 |
| S12F-CANDIDATE-03 | ORDINARY_PATH_ANALOG  |       50 |     -0.0430426 |                   0.951462 |                   165.243  |              0.195414 |           0.206094 |                0.176386 |                 -0.0190282  |              -1.73885  |          -0.318254 |                      0.138292  |
| S12F-CANDIDATE-03 | RECURRENCE_MAP_ANALOG |       50 |      0.199544  |                   0.564475 |                    66.1743 |              0.169179 |           0.206094 |                0.176386 |                  0.0072065  |             nan        |         nan        |                      0.103     |

### Predictor gates

| candidateId       | predictorId           |   spearmanQHat |   spearmanBootstrapLower95 |   brierImprovementOverPrior |   brierImprovementBootstrapLower95 | heldOutCommittorPredictionPassed   |
|:------------------|:----------------------|---------------:|---------------------------:|----------------------------:|-----------------------------------:|:-----------------------------------|
| S12F-CANDIDATE-02 | EXACT_H_TRACE_ANALOG  |      0.298928  |                  0.0348256 |                  0.00412448 |                        -0.00834934 | False                              |
| S12F-CANDIDATE-02 | OPERATOR_CHANGE       |      0.247945  |                 -0.0333039 |                 -0.00955181 |                        -0.0389852  | False                              |
| S12F-CANDIDATE-02 | ORDINARY_PATH_ANALOG  |      0.232078  |                 -0.0616471 |                 -0.00224496 |                        -0.0196962  | False                              |
| S12F-CANDIDATE-02 | RECURRENCE_MAP_ANALOG |      0.167309  |                 -0.153499  |                 -0.00361024 |                        -0.0254624  | False                              |
| S12F-CANDIDATE-03 | EXACT_H_TRACE_ANALOG  |      0.186226  |                 -0.0811265 |                  0.00346484 |                        -0.00881821 | False                              |
| S12F-CANDIDATE-03 | OPERATOR_CHANGE       |      0.33658   |                  0.0588208 |                 -0.00305832 |                        -0.0298755  | False                              |
| S12F-CANDIDATE-03 | ORDINARY_PATH_ANALOG  |     -0.0430426 |                 -0.355651  |                 -0.0190282  |                        -0.0401289  | False                              |
| S12F-CANDIDATE-03 | RECURRENCE_MAP_ANALOG |      0.199544  |                 -0.119294  |                  0.0072065  |                        -0.00782934 | False                              |

## Interpretation

This is an empirical finite-horizon entry probability conditioned on a retrospectively defined basin; it is not the classical infinite-horizon A-before-B committor and does not identify the paper authors' label. The original observed future contributes only the pre-existing target basin and a diagnostic single-future outcome. It does not drive branch dynamics, branch seeds, state selection or any frozen predictor. Catalytic matrix—not branch or molecular observation—is the higher-level inferential unit.

`STATE_DEPENDENT_COMMITTOR_ESTABLISHED` requires positive noise-corrected between-state variance with a bootstrap lower bound above zero, split-half Spearman above 0.5 with lower bound above 0.3, and at least 20 intermediate states in **each** simulator candidate. Failure stops the precursor-feature search under the human's conditional gate; it does not prove that no other target or horizon can have a committor.

## Runtime and provenance

- Repository lock: `252a43ccb072e3df200c870b00114b07210b2812`.
- CPU float64, `8` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `181.643`; aggregate worker CPU hours: `0.241614`.
- Source grounding: E & Vanden-Eijnden (2006), DOI `10.1007/s10955-005-9003-9`; Best & Hummer (2005), DOI `10.1073/pnas.0408098102`.

## Autonomous boundary

L28 is frozen. The existing authorization permits one next bounded loop, but no transition-tube/current analysis is eligible yet. S20, E02, author contact, interventions and report-bundle work remain inactive.
