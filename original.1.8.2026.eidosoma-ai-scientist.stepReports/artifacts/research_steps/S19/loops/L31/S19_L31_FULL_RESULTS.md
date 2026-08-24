# S19-L31 — Untouched Eight-Step Propagator Committor Confirmation

## Chief/human handoff

- **Step:** `E01-S19-L31-UNTOUCHED-EIGHT-STEP-PROPAGATOR-COMMITTOR-CONFIRMATION-v1.0.0`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** `UNTOUCHED_EIGHT_STEP_PROPAGATOR_COMMITTOR_COORDINATE_CONFIRMED`, `RETROSPECTIVE_BASIN_CONDITIONED_SHOOTING_SIGNAL`, `NOT_PROMOTABLE_AS_CONFIRMED_PAPER_RESULT`
- **Validation:** 80 deterministic states from candidate-specific matrices unused by L28–L30; exact state/beta/target restoration; 10,240 H32 and 5,120 original H8 branches plus common-stream target-reference controls; exact full branch/model replay; 4,096 matrix bootstraps; 512 label permutations; immutable-prior, seed, runtime/storage, regeneration and artifact gates passed.
- **Next bounded theme:** COMMITTOR_ORDERED_TRANSITION_TUBE_COORDINATE_DISCOVERY

## Frozen question and design

The unchanged candidate-specific L30 scaler, coefficients, H8 horizon, 64-branch estimator and nine input summaries were applied without refitting to eight unique unused matrices at each of five landmarks per candidate. The H32 response was independently re-estimated with 128 new branches per state. Branch dynamics use only the restored state and new streams; the target basin remains explicitly retrospective and completed-run matrix-specific.

## Untouched metrics

| referenceVariant             | candidateId       | modelId                       |   states |   spearmanQHat |   brierScorePerBranch |   binomialLogLossPerBranch |   calibrationIntercept |   calibrationSlope |
|:-----------------------------|:------------------|:------------------------------|---------:|---------------:|----------------------:|---------------------------:|-----------------------:|-------------------:|
| ORIGINAL                     | S12F-CANDIDATE-02 | DEVELOPMENT_PRIOR             |       40 |    nan         |              0.18624  |                   0.55957  |             nan        |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-02 | EIGHT_STEP_PROPAGATOR_MOMENTS |       40 |      0.900269  |              0.141955 |                   0.437431 |              -0.480451 |           0.80884  |
| ORIGINAL                     | S12F-CANDIDATE-02 | EXACT_H_TRACE_ANALOG          |       40 |      0.212663  |              0.18549  |                   0.555025 |             nan        |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-02 | ORDINARY_PATH_ANALOG          |       40 |      0.418784  |              0.183117 |                   0.550552 |              -0.332261 |           0.561825 |
| ORIGINAL                     | S12F-CANDIDATE-02 | Q8_CALIBRATED                 |       40 |      0.766359  |              0.143566 |                   0.455184 |               0.349604 |           1.17413  |
| ORIGINAL                     | S12F-CANDIDATE-02 | Q8_JEFFREYS_DIRECT            |       40 |      0.766359  |              0.2005   |                   0.743653 |               1.94195  |           0.852912 |
| ORIGINAL                     | S12F-CANDIDATE-02 | TARGET_GEOMETRY_CONTROL       |       40 |      0.625265  |              0.186494 |                   0.565548 |              -0.538017 |           0.436327 |
| ORIGINAL                     | S12F-CANDIDATE-03 | DEVELOPMENT_PRIOR             |       40 |    nan         |              0.229381 |                   0.659613 |               0.207705 |           0.719859 |
| ORIGINAL                     | S12F-CANDIDATE-03 | EIGHT_STEP_PROPAGATOR_MOMENTS |       40 |      0.821161  |              0.118847 |                   0.401198 |              -0.181684 |           0.722082 |
| ORIGINAL                     | S12F-CANDIDATE-03 | EXACT_H_TRACE_ANALOG          |       40 |      0.146592  |              0.228632 |                   0.691707 |             nan        |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-03 | ORDINARY_PATH_ANALOG          |       40 |      0.266459  |              0.22634  |                   0.667225 |             nan        |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-03 | Q8_CALIBRATED                 |       40 |      0.848893  |              0.107692 |                   0.358806 |             nan        |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-03 | Q8_JEFFREYS_DIRECT            |       40 |      0.848893  |              0.148407 |                   0.543736 |             nan        |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-03 | TARGET_GEOMETRY_CONTROL       |       40 |      0.544029  |              0.20297  |                   0.611269 |             nan        |         nan        |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | EIGHT_STEP_PROPAGATOR_MOMENTS |       40 |      0.223047  |              0.222322 |                   0.818634 |              -0.748289 |           0.119024 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | Q8_CALIBRATED                 |       40 |     -0.0694668 |              0.207573 |                   0.64648  |              -3.78967  |          -1.24025  |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | Q8_JEFFREYS_DIRECT            |       40 |     -0.0694668 |              0.243641 |                   1.20768  |             nan        |         nan        |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | EIGHT_STEP_PROPAGATOR_MOMENTS |       40 |     -0.0379272 |              0.309043 |                   1.38633  |             nan        |         nan        |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | Q8_CALIBRATED                 |       40 |    nan         |              0.282691 |                   0.930448 |             nan        |         nan        |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | Q8_JEFFREYS_DIRECT            |       40 |    nan         |              0.313136 |                   1.55299  |             nan        |         nan        |

## Confirmation gates

| candidateId       |   states |   correctedBetweenStateVariance |   correctedVarianceLower95 |   q32SplitHalfSpearman |   q32SplitHalfLower95 |   intermediateStateCount |   h8SplitHalfSpearman |   h8SplitHalfLower95 |   primarySpearman |   primarySpearmanLower95 |   brierImprovementLowerVsDEVELOPMENT_PRIOR |   brierImprovementLowerVsTARGET_GEOMETRY_CONTROL |   brierImprovementLowerVsEXACT_H_TRACE_ANALOG |   brierImprovementLowerVsORDINARY_PATH_ANALOG |   labelPermutationP |   targetReferenceSpearman | correctedVariancePassed   | q32ReliabilityPassed   | intermediateSupportPassed   | h8ReliabilityPassed   | primaryRankPassed   | incrementalBrierPassed   | labelPermutationPassed   | targetReferenceControlPassed   | candidateConfirmationGatePassed   |
|:------------------|---------:|--------------------------------:|---------------------------:|-----------------------:|----------------------:|-------------------------:|----------------------:|---------------------:|------------------:|-------------------------:|-------------------------------------------:|-------------------------------------------------:|----------------------------------------------:|----------------------------------------------:|--------------------:|--------------------------:|:--------------------------|:-----------------------|:----------------------------|:----------------------|:--------------------|:-------------------------|:-------------------------|:-------------------------------|:----------------------------------|
| S12F-CANDIDATE-02 |       40 |                       0.0716707 |                  0.0333022 |               0.930238 |              0.82106  |                       26 |              0.686025 |             0.391786 |          0.900269 |                 0.79113  |                                 0.00978239 |                                       0.00643895 |                                    0.00665599 |                                    0.00472605 |          0.00194932 |                 0.223047  | True                      | True                   | True                        | True                  | True                | True                     | True                     | True                           | True                              |
| S12F-CANDIDATE-03 |       40 |                       0.120689  |                  0.0731012 |               0.958141 |              0.890853 |                       20 |              0.931886 |             0.808835 |          0.821161 |                 0.631763 |                                 0.0484238  |                                       0.0389088  |                                    0.0482454  |                                    0.0431624  |          0.00194932 |                -0.0379272 | True                      | True                   | True                        | True                  | True                | True                     | True                     | True                           | True                              |

## Interpretation boundary

A passing result confirms a simulation-accessible finite-horizon shooting coordinate for this reconstructed retrospective basin. It is not a directly observed biomarker, a prospective author-label result, causal control, or author-code identification. It can license the next bounded attempt to distill a path/tube coordinate, but not a causal-current claim by itself.

## Runtime

- Repository lock: `28ae031d27f3272e6f8f4021aae087bf9b2590fd`.
- CPU float64, `8` workers, no GPU.
- Wall seconds: `175.571`; estimated worker CPU hours: `0.149133`.

## Autonomous boundary

L31 is frozen. S20, E02, author contact, interventions and report-bundle work remain inactive.
