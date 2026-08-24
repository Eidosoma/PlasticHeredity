# S19-L29 — Exact GARD Generator Drift/Diffusion Committor Coordinate

## Chief/human handoff

- **Step:** `E01-S19-L29-EXACT-GARD-GENERATOR-COMMITTOR-COORDINATE-v1.0.0`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** `EXACT_GARD_GENERATOR_FEATURES_MISS_COMMITTOR_SIGNAL`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** exact L28 state/beta/target/q identities; source-defined analytic moments; 2048 complete one-step kernel samples per state; independent moment halves; full feature and model replay; development-only fit; 4,096 matrix bootstraps; 512 development-label permutations; reference, seed, immutable-prior, runtime/storage and artifact gates passed.
- **Next bounded theme:** MULTISTEP_GENERATOR_PROPAGATOR_OR_MEMORY_STATE_AUDIT

## Frozen question

Do exact local GARD birth/death/fission drift and diffusion features recover the reliable L28 H32 committor on held-out matrices beyond target geometry and prior exact-H/ordinary representations?

## Method boundary

The analytical branch uses source-defined clipped-Poisson pre-trim growth moments and exact candidate-specific fission moments. The complete-kernel branch estimates the implemented one-selected-clock transition moments with 2,048 independent samples, including overshoot trim and daughter selection. It generates no new H32 future and does not reuse any L28 branch stream. Target-radial features are explicitly conditioned on the completed-run matrix-specific basin; they are retrospective-basin-conditioned and cannot establish online early warning.

## Held-out validation metrics

| referenceVariant             | candidateId       | modelId                          |   states |   spearmanQHat |   brierScorePerBranch |   binomialLogLossPerBranch |   calibrationIntercept |   calibrationSlope |
|:-----------------------------|:------------------|:---------------------------------|---------:|---------------:|----------------------:|---------------------------:|-----------------------:|-------------------:|
| ORIGINAL                     | S12F-CANDIDATE-02 | ANALYTIC_RADIAL_GENERATOR        |       50 |      0.5254    |              0.150784 |                   0.525152 |              -0.459381 |          0.477773  |
| ORIGINAL                     | S12F-CANDIDATE-02 | BASIN_BLIND_GENERATOR            |       50 |      0.378911  |              0.233683 |                   0.936394 |             nan        |        nan         |
| ORIGINAL                     | S12F-CANDIDATE-02 | COMPLETE_KERNEL_RADIAL_GENERATOR |       50 |      0.554045  |              0.153242 |                   0.510741 |              -0.452804 |          0.521355  |
| ORIGINAL                     | S12F-CANDIDATE-02 | DEVELOPMENT_PRIOR                |       50 |    nan         |              0.188434 |                   0.564386 |             nan        |        nan         |
| ORIGINAL                     | S12F-CANDIDATE-02 | EXACT_H_TRACE_ANALOG             |       50 |      0.298928  |              0.18431  |                   0.551608 |             nan        |        nan         |
| ORIGINAL                     | S12F-CANDIDATE-02 | OPERATOR_CHANGE                  |       50 |      0.247945  |              0.197986 |                   0.583857 |              -0.746533 |          0.416935  |
| ORIGINAL                     | S12F-CANDIDATE-02 | ORDINARY_PATH_ANALOG             |       50 |      0.232078  |              0.190679 |                   0.591805 |              -0.748541 |          0.22373   |
| ORIGINAL                     | S12F-CANDIDATE-02 | RECURRENCE_MAP_ANALOG            |       50 |      0.167309  |              0.192044 |                   0.726656 |             nan        |        nan         |
| ORIGINAL                     | S12F-CANDIDATE-02 | TARGET_GEOMETRY_CONTROL          |       50 |      0.513337  |              0.177961 |                   0.544779 |             nan        |        nan         |
| ORIGINAL                     | S12F-CANDIDATE-03 | ANALYTIC_RADIAL_GENERATOR        |       50 |      0.652577  |              0.131317 |                   0.440601 |              -0.509175 |          0.539253  |
| ORIGINAL                     | S12F-CANDIDATE-03 | BASIN_BLIND_GENERATOR            |       50 |      0.433336  |              0.185874 |                   0.570753 |             nan        |        nan         |
| ORIGINAL                     | S12F-CANDIDATE-03 | COMPLETE_KERNEL_RADIAL_GENERATOR |       50 |      0.679555  |              0.134904 |                   0.448388 |              -0.501477 |          0.530817  |
| ORIGINAL                     | S12F-CANDIDATE-03 | DEVELOPMENT_PRIOR                |       50 |    nan         |              0.176386 |                   0.538026 |             nan        |        nan         |
| ORIGINAL                     | S12F-CANDIDATE-03 | EXACT_H_TRACE_ANALOG             |       50 |      0.186226  |              0.172921 |                   0.594398 |              -0.944838 |          0.162511  |
| ORIGINAL                     | S12F-CANDIDATE-03 | OPERATOR_CHANGE                  |       50 |      0.33658   |              0.179444 |                   0.538851 |             nan        |        nan         |
| ORIGINAL                     | S12F-CANDIDATE-03 | ORDINARY_PATH_ANALOG             |       50 |     -0.0430426 |              0.195414 |                   0.951462 |              -1.73885  |         -0.318254  |
| ORIGINAL                     | S12F-CANDIDATE-03 | RECURRENCE_MAP_ANALOG            |       50 |      0.199544  |              0.169179 |                   0.564475 |             nan        |        nan         |
| ORIGINAL                     | S12F-CANDIDATE-03 | TARGET_GEOMETRY_CONTROL          |       50 |      0.654453  |              0.149441 |                   0.473713 |              -0.390311 |          0.584306  |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | ANALYTIC_RADIAL_GENERATOR        |       50 |      0.243236  |              0.275636 |                   1.23393  |              -1.08819  |          0.0446016 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | BASIN_BLIND_GENERATOR            |       50 |      0.378911  |              0.233683 |                   0.936394 |             nan        |        nan         |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | COMPLETE_KERNEL_RADIAL_GENERATOR |       50 |      0.263277  |              0.264292 |                   1.15045  |             nan        |        nan         |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | DEVELOPMENT_PRIOR                |       50 |    nan         |              0.188434 |                   0.564386 |             nan        |        nan         |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | TARGET_GEOMETRY_CONTROL          |       50 |      0.198251  |              0.221843 |                   0.664729 |              -0.993465 |          0.08619   |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | ANALYTIC_RADIAL_GENERATOR        |       50 |      0.438915  |              0.156193 |                   0.492929 |              -0.568557 |          0.519776  |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | BASIN_BLIND_GENERATOR            |       50 |      0.433336  |              0.185874 |                   0.570753 |             nan        |        nan         |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | COMPLETE_KERNEL_RADIAL_GENERATOR |       50 |      0.38366   |              0.152101 |                   0.489059 |             nan        |        nan         |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | DEVELOPMENT_PRIOR                |       50 |    nan         |              0.176386 |                   0.538026 |             nan        |        nan         |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | TARGET_GEOMETRY_CONTROL          |       50 |      0.219145  |              0.178979 |                   0.55781  |             nan        |        nan         |

## One-step moment reliability

| referenceVariant             | candidateId       | feature        |   splitHalfSpearman |   meanAbsoluteDifference |
|:-----------------------------|:------------------|:---------------|--------------------:|-------------------------:|
| ORIGINAL                     | S12F-CANDIDATE-02 | MuNorm         |            0.996844 |              0.00053028  |
| ORIGINAL                     | S12F-CANDIDATE-02 | DiffusionTrace |            0.995452 |              4.82243e-05 |
| ORIGINAL                     | S12F-CANDIDATE-02 | ScoreDrift     |            0.995272 |              0.000839282 |
| ORIGINAL                     | S12F-CANDIDATE-02 | ScoreVariance  |            0.996616 |              6.54335e-05 |
| ORIGINAL                     | S12F-CANDIDATE-02 | BrownianHit32  |            0.996918 |              0.00403905  |
| ORIGINAL                     | S12F-CANDIDATE-03 | MuNorm         |            0.997096 |              0.000516891 |
| ORIGINAL                     | S12F-CANDIDATE-03 | DiffusionTrace |            0.997888 |              3.73168e-05 |
| ORIGINAL                     | S12F-CANDIDATE-03 | ScoreDrift     |            0.992763 |              0.000779166 |
| ORIGINAL                     | S12F-CANDIDATE-03 | ScoreVariance  |            0.991935 |              5.00813e-05 |
| ORIGINAL                     | S12F-CANDIDATE-03 | BrownianHit32  |            0.994539 |              0.00544936  |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | MuNorm         |            0.996844 |              0.00053028  |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | DiffusionTrace |            0.995452 |              4.82243e-05 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | ScoreDrift     |            0.981734 |              0.000413241 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | ScoreVariance  |            0.960144 |              2.01559e-05 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | BrownianHit32  |            0.963586 |              0.00248074  |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | MuNorm         |            0.997096 |              0.000516891 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | DiffusionTrace |            0.997888 |              3.73168e-05 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | ScoreDrift     |            0.989511 |              0.000435609 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | ScoreVariance  |            0.928737 |              1.90306e-05 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | BrownianHit32  |            0.929798 |              0.000195421 |

## Gate adjudication

| candidateId       |   primarySpearman |   primarySpearmanBootstrapLower95 |   brierImprovementLowerVsTARGET_GEOMETRY_CONTROL |   brierImprovementLowerVsBASIN_BLIND_GENERATOR |   brierImprovementLowerVsEXACT_H_TRACE_ANALOG |   brierImprovementLowerVsORDINARY_PATH_ANALOG |   developmentPermutationP | rankGatePassed   | allIncrementalBrierGatesPassed   | developmentPermutationPassed   | exactReplayPassed   | candidateCoordinateGatePassed   |
|:------------------|------------------:|----------------------------------:|-------------------------------------------------:|-----------------------------------------------:|----------------------------------------------:|----------------------------------------------:|--------------------------:|:-----------------|:---------------------------------|:-------------------------------|:--------------------|:--------------------------------|
| S12F-CANDIDATE-02 |          0.554045 |                          0.320643 |                                      -0.00640223 |                                      0.0167381 |                                   -0.0118368  |                                    -0.0179671 |                0.00389864 | True             | False                            | True                           | True                | False                           |
| S12F-CANDIDATE-03 |          0.679555 |                          0.480921 |                                      -0.00538334 |                                      0.0135004 |                                    0.00698849 |                                     0.011627  |                0.00194932 | True             | False                            | True                           | True                | False                           |

## Interpretation

A local generator coordinate must rank held-out q with Spearman above 0.5 and bootstrap lower bound above 0.3, improve Brier score beyond the development prior, target geometry, basin-blind generator, frozen exact-H trace and ordinary path with lower bounds above zero, and pass the development-label permutation in both candidates. A target-conditioned pass remains retrospective; a basin-blind pass would be the stronger online-state clue. No result is confirmatory without a later untouched seed-firewalled cohort.

## Runtime and provenance

- Repository lock: `97bc2a67e526ddd12d059d60ab8a7016526598ea`.
- CPU float64, `8` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `129.534`; aggregate worker CPU hours: `0.013404`.
- Method grounding: Gillespie, *The Chemical Langevin Equation*, DOI `10.1063/1.481811`; L28 finite-horizon shooting audit.

## Autonomous boundary

L29 is frozen. Transition-tube/reactive-current work remains prohibited unless this loop establishes a held-out committor-predictive coordinate in both candidates. S20, E02, author contact, interventions and report-bundle work remain inactive.
