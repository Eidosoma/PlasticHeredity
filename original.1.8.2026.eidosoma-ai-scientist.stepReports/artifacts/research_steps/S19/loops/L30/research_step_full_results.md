# S19-L30 — Eight-Step Generator Propagator Committor Coordinate

## Chief/human handoff

- **Step:** `E01-S19-L30-EIGHT-STEP-GENERATOR-PROPAGATOR-COMMITTOR-COORDINATE-v1.0.0`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** `EIGHT_STEP_PROPAGATOR_COMMITTOR_COORDINATE_ESTABLISHED`, `RETROSPECTIVE_BASIN_CONDITIONED_SHOOTING_SIGNAL`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** exact L28 state/target/q replay; 12,800 new domain-separated H8 branches; independent 32/32 halves; full original-branch and model replay; target-reference and development-label controls; 4,096 matrix bootstraps; immutable-prior, runtime/storage, regeneration and artifact hashes passed.
- **Next bounded theme:** UNTOUCHED_SHORT_PROPAGATOR_COORDINATE_CONFIRMATION

## Frozen question and method

This loop asks whether the one-step L29 signal becomes a stable H32 committor coordinate after exactly eight selected-clock observations. It uses 64 new independent short futures per restored L28 state and never propagates the predictor to H32. The target basin remains retrospectively completed-run conditioned. The primary coordinate is calibrated on development matrices only and evaluated unchanged on validation matrices.

## Held-out metrics

| referenceVariant             | candidateId       | modelId                       |   states |   spearmanQHat |   brierScorePerBranch |   binomialLogLossPerBranch |   calibrationIntercept |   calibrationSlope |
|:-----------------------------|:------------------|:------------------------------|---------:|---------------:|----------------------:|---------------------------:|-----------------------:|-------------------:|
| ORIGINAL                     | S12F-CANDIDATE-02 | DEVELOPMENT_PRIOR             |       50 |    nan         |              0.188434 |                   0.564386 |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-02 | EIGHT_STEP_PROPAGATOR_MOMENTS |       50 |      0.771279  |              0.105435 |                   0.352218 |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-02 | EXACT_H_TRACE_ANALOG          |       50 |      0.298928  |              0.18431  |                   0.551608 |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-02 | OPERATOR_CHANGE               |       50 |      0.247945  |              0.197986 |                   0.583857 |           -0.746533    |           0.416935 |
| ORIGINAL                     | S12F-CANDIDATE-02 | ORDINARY_PATH_ANALOG          |       50 |      0.232078  |              0.190679 |                   0.591805 |           -0.748541    |           0.22373  |
| ORIGINAL                     | S12F-CANDIDATE-02 | Q8_CALIBRATED                 |       50 |      0.743224  |              0.105231 |                   0.361361 |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-02 | Q8_JEFFREYS_DIRECT            |       50 |      0.743224  |              0.149373 |                   0.557302 |            2.22079     |           0.987403 |
| ORIGINAL                     | S12F-CANDIDATE-02 | RECURRENCE_MAP_ANALOG         |       50 |      0.167309  |              0.192044 |                   0.726656 |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-02 | TARGET_GEOMETRY_CONTROL       |       50 |      0.513337  |              0.177961 |                   0.544779 |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-03 | DEVELOPMENT_PRIOR             |       50 |    nan         |              0.176386 |                   0.538026 |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-03 | EIGHT_STEP_PROPAGATOR_MOMENTS |       50 |      0.887495  |              0.11133  |                   0.35339  |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-03 | EXACT_H_TRACE_ANALOG          |       50 |      0.186226  |              0.172921 |                   0.594398 |           -0.944838    |           0.162511 |
| ORIGINAL                     | S12F-CANDIDATE-03 | OPERATOR_CHANGE               |       50 |      0.33658   |              0.179444 |                   0.538851 |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-03 | ORDINARY_PATH_ANALOG          |       50 |     -0.0430426 |              0.195414 |                   0.951462 |           -1.73885     |          -0.318254 |
| ORIGINAL                     | S12F-CANDIDATE-03 | Q8_CALIBRATED                 |       50 |      0.754172  |              0.114369 |                   0.375514 |           -0.259025    |           0.774817 |
| ORIGINAL                     | S12F-CANDIDATE-03 | Q8_JEFFREYS_DIRECT            |       50 |      0.754172  |              0.133567 |                   0.517837 |            1.11683     |           0.717873 |
| ORIGINAL                     | S12F-CANDIDATE-03 | RECURRENCE_MAP_ANALOG         |       50 |      0.199544  |              0.169179 |                   0.564475 |          nan           |         nan        |
| ORIGINAL                     | S12F-CANDIDATE-03 | TARGET_GEOMETRY_CONTROL       |       50 |      0.654453  |              0.149441 |                   0.473713 |           -0.390311    |           0.584306 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | DEVELOPMENT_PRIOR             |       50 |    nan         |              0.188434 |                   0.564386 |          nan           |         nan        |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | EIGHT_STEP_PROPAGATOR_MOMENTS |       50 |      0.279473  |              0.185244 |                   0.552205 |           -0.513969    |           0.612543 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | Q8_CALIBRATED                 |       50 |    nan         |              0.188434 |                   0.564386 |            0.000979881 |           0.998922 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 | Q8_JEFFREYS_DIRECT            |       50 |    nan         |              0.248059 |                   1.23179  |          nan           |         nan        |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | DEVELOPMENT_PRIOR             |       50 |    nan         |              0.176386 |                   0.538026 |          nan           |         nan        |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | EIGHT_STEP_PROPAGATOR_MOMENTS |       50 |      0.291375  |              0.179748 |                   0.561803 |          nan           |         nan        |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | Q8_CALIBRATED                 |       50 |    nan         |              0.176385 |                   0.538022 |          nan           |         nan        |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 | Q8_JEFFREYS_DIRECT            |       50 |    nan         |              0.224367 |                   1.11485  |          nan           |         nan        |

## Short-propagator split-half reliability

| referenceVariant             | candidateId       |   states |   splitHalfSpearman |   q8VsQ32Spearman |    meanQ8 |   meanQ32 |   zeroQ8States |
|:-----------------------------|:------------------|---------:|--------------------:|------------------:|----------:|----------:|---------------:|
| ORIGINAL                     | S12F-CANDIDATE-02 |      100 |            0.818588 |          0.739446 | 0.090625  |  0.251641 |             64 |
| ORIGINAL                     | S12F-CANDIDATE-03 |      100 |            0.864741 |          0.741203 | 0.0932812 |  0.216953 |             67 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-02 |      100 |          nan        |        nan        | 0         |  0.251641 |            100 |
| TARGET_REFERENCE_PERMUTATION | S12F-CANDIDATE-03 |      100 |          nan        |        nan        | 0         |  0.216953 |            100 |

## Gate adjudication

| candidateId       |   primarySpearman |   spearmanBootstrapLower95 |   brierImprovementLowerVsDEVELOPMENT_PRIOR |   brierImprovementLowerVsTARGET_GEOMETRY_CONTROL |   brierImprovementLowerVsEXACT_H_TRACE_ANALOG |   brierImprovementLowerVsORDINARY_PATH_ANALOG |   developmentPermutationP |   targetPermutedSpearman | rankPassed   | incrementalBrierPassed   | permutationPassed   | targetReferenceControlPassed   | candidateCoordinateGatePassed   |
|:------------------|------------------:|---------------------------:|-------------------------------------------:|-------------------------------------------------:|----------------------------------------------:|----------------------------------------------:|--------------------------:|-------------------------:|:-------------|:-------------------------|:--------------------|:-------------------------------|:--------------------------------|
| S12F-CANDIDATE-02 |          0.771279 |                   0.590282 |                                  0.0377478 |                                        0.0278843 |                                     0.0303448 |                                     0.0373138 |                0.00194932 |                 0.279473 | True         | True                     | True                | True                           | True                            |
| S12F-CANDIDATE-03 |          0.887495 |                   0.797657 |                                  0.0264874 |                                        0.0115175 |                                     0.0286458 |                                     0.0335212 |                0.00194932 |                 0.291375 | True         | True                     | True                | True                           | True                            |

## Interpretation boundary

Even a passing short-propagator coordinate would be a simulation-based, retrospective-basin-conditioned reaction coordinate—not an observed early-warning biomarker, author-code reconstruction, or causal control result. It would require untouched state/matrix confirmation before any transition-current analysis. A failure would redirect the search to hidden-memory/history representations rather than horizon or branch-count tuning.

## Runtime and provenance

- Repository lock: `bc061662eca4fd468edbe9555b22ad25566330c2`.
- CPU float64, `8` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `117.190`; aggregate worker CPU hours: `0.046865`.

## Autonomous boundary

L30 is frozen. S20, E02, author contact, interventions and report-bundle work remain inactive.
