# S19-L41 — Fission-Clock Repeated Cross-Generation Recurrence

## Chief/human handoff

- **Step:** `E01-S19-L41-FISSION-CLOCK-REPEATED-CROSS-GENERATION-RECURRENCE-v1.0.0`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** `NO_RELIABLE_REPEATED_RECURRENCE_COMMITTOR_AT_F12`, `PROCESS_TARGET_REQUIRES_REDEFINITION`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** immutable L40-and-earlier baseline; seven fixtures; zero-overlap branch and analysis seeds; 53,760 independent F12/F4 branch streams; candidate-separated 4,096 matrix bootstraps; exact full branch, score, statistic and report regeneration; storage and artifact hashes.
- **Recommended next action:** `FISSION_CONDITIONED_HEREDITY_RECOVERY_HAZARD`.

## Frozen question

Does a clock matched to the process reveal a reliable probability of repeated recurrence? A return is certified only when a future post-fission state has strict `H>0.9` to an eligible boundary at least two generations earlier and the immediately preceding boundary was `H<=0.9` to that same reference. At most one return is counted per future boundary. The primary event is online certification of a second return within exactly 12 future fissions. Continuous residence near a reference does not count as repeated recovery.

The short coordinate is calculated from an independent 64-branch, four-fission ensemble. It is not a static biomarker. No completed trajectory, completed-run centroid, threshold search, recurrence-count search, horizon search, paper label, emergence value, intervention or new catalytic matrix enters this loop.

## Anchor results

### F12 process probabilities

| evaluationCohort   | candidateId       |   states |    meanQ |   meanAnyReturn |   meanMembershipOpportunity |   meanReturnBoundaries |
|:-------------------|:------------------|---------:|---------:|----------------:|----------------------------:|-----------------------:|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 |       50 | 0.136719 |        0.307656 |                    0.315312 |               0.54     |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 |       50 | 0.165625 |        0.347031 |                    0.370781 |               0.625    |
| L28_VALIDATION     | S12F-CANDIDATE-02 |       50 | 0.23625  |        0.419219 |                    0.447031 |               0.993281 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |       50 | 0.156406 |        0.350781 |                    0.379531 |               0.657656 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |       40 | 0.186719 |        0.379492 |                    0.425586 |               0.759766 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |       40 | 0.175586 |        0.351172 |                    0.382617 |               0.754883 |

### Empirical committor reliability

| evaluationCohort   | candidateId       | branchFamily   | targetId               |   states |   eligibleStates |    meanQ |   minimumQ |   maximumQ |   intermediateStateCount |   observedBetweenStateVariance |   estimatedBinomialNoiseVariance |   correctedBetweenStateVariance |   correctedVarianceLower95 |   correctedVarianceUpper95 |   splitHalfSpearman |   splitHalfLower95 |   splitHalfUpper95 | reliabilityGatePassed   |
|:-------------------|:------------------|:---------------|:-----------------------|---------:|-----------------:|---------:|-----------:|-----------:|-------------------------:|-------------------------------:|---------------------------------:|--------------------------------:|---------------------------:|---------------------------:|--------------------:|-------------------:|-------------------:|:------------------------|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12            | PRIMARY_PREFIX_HISTORY |       50 |               50 | 0.136719 |          0 |   0.914062 |                       17 |                      0.036552  |                      0.00064729  |                       0.0359047 |                  0.0107349 |                  0.0660712 |            0.86474  |           0.751091 |           0.935323 | False                   |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12            | PRIMARY_PREFIX_HISTORY |       50 |               50 | 0.165625 |          0 |   0.632812 |                       23 |                      0.0353954 |                      0.000815007 |                       0.0345804 |                  0.0208288 |                  0.0469869 |            0.923927 |           0.84218  |           0.965479 | True                    |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12            | PRIMARY_PREFIX_HISTORY |       50 |               50 | 0.23625  |          0 |   0.992188 |                       19 |                      0.0847605 |                      0.000766698 |                       0.0839938 |                  0.0503014 |                  0.114764  |            0.936674 |           0.85765  |           0.976428 | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12            | PRIMARY_PREFIX_HISTORY |       50 |               50 | 0.156406 |          0 |   0.828125 |                       23 |                      0.035713  |                      0.000763343 |                       0.0349497 |                  0.0159368 |                  0.0573436 |            0.874691 |           0.748619 |           0.940684 | True                    |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12            | PRIMARY_PREFIX_HISTORY |       40 |               40 | 0.186719 |          0 |   0.96875  |                       21 |                      0.0424642 |                      0.000869703 |                       0.0415945 |                  0.0184295 |                  0.0730841 |            0.928858 |           0.838632 |           0.970436 | True                    |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12            | PRIMARY_PREFIX_HISTORY |       40 |               40 | 0.175586 |          0 |   1        |                       17 |                      0.0518423 |                      0.000741805 |                       0.0511005 |                  0.0158041 |                  0.0910156 |            0.885009 |           0.750549 |           0.95217  | False                   |

### Short shooting, reference, order and inheritance controls

| evaluationCohort   | candidateId       | comparisonId                        | comparisonType   |   definedPairs |   pointEstimate |    lower95 |    upper95 | gatePassed   |
|:-------------------|:------------------|:------------------------------------|:-----------------|---------------:|----------------:|-----------:|-----------:|:-------------|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12_INHERITANCE_FRACTION_VS_PRIMARY | RANK             |             50 |       0.873516  |  0.772094  |  0.92966   | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_ORDER_PERMUTED    | DIFFERENCE       |             50 |      -0.432344  | -0.483281  | -0.381406  | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_SPECIES_PERMUTED  | DIFFERENCE       |             50 |       0.0457813 |  0.0248438 |  0.0715625 | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_UNRELATED         | DIFFERENCE       |             50 |       0.045625  |  0.0248438 |  0.0720313 | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F4_RETURN_COUNT_VS_F12_PRIMARY      | RANK             |             50 |       0.779781  |  0.648189  |  0.868367  | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12_INHERITANCE_FRACTION_VS_PRIMARY | RANK             |             50 |       0.834518  |  0.686547  |  0.914184  | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_ORDER_PERMUTED    | DIFFERENCE       |             50 |      -0.434844  | -0.484844  | -0.383242  | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_SPECIES_PERMUTED  | DIFFERENCE       |             50 |       0.0659375 |  0.0397461 |  0.0951562 | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_UNRELATED         | DIFFERENCE       |             50 |       0.0664062 |  0.0399023 |  0.0971875 | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F4_RETURN_COUNT_VS_F12_PRIMARY      | RANK             |             50 |       0.728995  |  0.522243  |  0.870493  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12_INHERITANCE_FRACTION_VS_PRIMARY | RANK             |             50 |       0.705674  |  0.506232  |  0.828904  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_ORDER_PERMUTED    | DIFFERENCE       |             50 |      -0.420938  | -0.478066  | -0.364746  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_SPECIES_PERMUTED  | DIFFERENCE       |             50 |       0.104844  |  0.0596875 |  0.157852  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_UNRELATED         | DIFFERENCE       |             50 |       0.112969  |  0.0667187 |  0.166699  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F4_RETURN_COUNT_VS_F12_PRIMARY      | RANK             |             50 |       0.806498  |  0.650081  |  0.89766   | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12_INHERITANCE_FRACTION_VS_PRIMARY | RANK             |             50 |       0.648054  |  0.392962  |  0.811717  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_ORDER_PERMUTED    | DIFFERENCE       |             50 |      -0.480156  | -0.521563  | -0.437344  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_SPECIES_PERMUTED  | DIFFERENCE       |             50 |       0.0754688 |  0.0459375 |  0.110312  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_UNRELATED         | DIFFERENCE       |             50 |       0.075625  |  0.045625  |  0.10875   | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F4_RETURN_COUNT_VS_F12_PRIMARY      | RANK             |             50 |       0.730417  |  0.521154  |  0.874213  | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12_INHERITANCE_FRACTION_VS_PRIMARY | RANK             |             40 |       0.728904  |  0.502027  |  0.875831  | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_ORDER_PERMUTED    | DIFFERENCE       |             40 |      -0.450781  | -0.499731  | -0.396484  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_SPECIES_PERMUTED  | DIFFERENCE       |             40 |       0.0878906 |  0.0473389 |  0.13418   | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_UNRELATED         | DIFFERENCE       |             40 |       0.0859375 |  0.0462891 |  0.133008  | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F4_RETURN_COUNT_VS_F12_PRIMARY      | RANK             |             40 |       0.811564  |  0.673235  |  0.888684  | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12_INHERITANCE_FRACTION_VS_PRIMARY | RANK             |             40 |       0.841073  |  0.695974  |  0.917183  | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_ORDER_PERMUTED    | DIFFERENCE       |             40 |      -0.438281  | -0.496289  | -0.374805  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_SPECIES_PERMUTED  | DIFFERENCE       |             40 |       0.0625    |  0.0314453 |  0.0990234 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_UNRELATED         | DIFFERENCE       |             40 |       0.0568359 |  0.0250732 |  0.0942627 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F4_RETURN_COUNT_VS_F12_PRIMARY      | RANK             |             40 |       0.777061  |  0.593285  |  0.880707  | True         |

### Renewal and ordinary-inheritance context

| evaluationCohort   | candidateId       |   branches |   meanAnyReturnProbability |   meanRepeatedReturnProbability |   meanMembershipOnlyProbability |   meanFirstReturnBoundary |   meanCertificationBoundary |   meanInterReturnGap |   meanInheritanceFraction |   meanMaximumInheritanceRun |   completedHorizonFraction | indexLookupExact   |
|:-------------------|:------------------|-----------:|---------------------------:|--------------------------------:|--------------------------------:|--------------------------:|----------------------------:|---------------------:|--------------------------:|----------------------------:|---------------------------:|:-------------------|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 |       6400 |                   0.307656 |                        0.136719 |                        0.315312 |                   7.03149 |                     8.77257 |              2.86057 |                  0.811523 |                     7.72844 |                          1 | True               |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 |       6400 |                   0.347031 |                        0.165625 |                        0.370781 |                   7.11391 |                     8.72642 |              2.63113 |                  0.822214 |                     7.92281 |                          1 | True               |
| L28_VALIDATION     | S12F-CANDIDATE-02 |       6400 |                   0.419219 |                        0.23625  |                        0.447031 |                   5.85203 |                     7.08135 |              2.53704 |                  0.845911 |                     8.40422 |                          1 | True               |
| L28_VALIDATION     | S12F-CANDIDATE-03 |       6400 |                   0.350781 |                        0.156406 |                        0.379531 |                   6.88864 |                     8.2008  |              2.84715 |                  0.848867 |                     8.32734 |                          1 | True               |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |       5120 |                   0.379492 |                        0.186719 |                        0.425586 |                   7.0983  |                     8.40272 |              2.56904 |                  0.838607 |                     8.2543  |                          1 | True               |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |       5120 |                   0.351172 |                        0.175586 |                        0.382617 |                   6.86541 |                     8.24805 |              2.86874 |                  0.808594 |                     7.84258 |                          1 | True               |

### Locked scientific gates

| evaluationCohort   | candidateId       | primaryF12Reliable   | speciesPermutationControlPassed   | unrelatedMatrixControlPassed   | futureOrderControlPassed   |   membershipOpportunityMinusPrimary | membershipOpportunityGatePassed   | repeatedRecurrenceTargetPassed   | f4ReturnCountRankPassed   | shortShootingCoordinatePassed   |
|:-------------------|:------------------|:---------------------|:----------------------------------|:-------------------------------|:---------------------------|------------------------------------:|:----------------------------------|:---------------------------------|:--------------------------|:--------------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | False                | True                              | True                           | False                      |                            0.210781 | True                              | False                            | True                      | False                           |
| L28_VALIDATION     | S12F-CANDIDATE-03 | True                 | True                              | True                           | False                      |                            0.223125 | True                              | False                            | True                      | False                           |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | True                 | True                              | True                           | False                      |                            0.238867 | True                              | False                            | True                      | False                           |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | False                | True                              | True                           | False                      |                            0.207031 | True                              | False                            | True                      | False                           |

## Interpretation

The event separates three objects that previous loops partially conflated: ordinary parent-to-daughter inheritance frequency, temporal ordering of compositional membership, and recovery after a genuine far-to-near transition. The ordinary inheritance fraction and run length remain controls; neither defines the target. Online certification remains distinct from the first return and from any retrospective physical onset.

The one-per-branch future-order control preserves the sampled future boundary compositions and fission count while changing their temporal order. Species-permuted and unrelated prefixes test whether apparent recovery is specific to the observed past. The membership-only event quantifies how often the same compositions would look recurrent without requiring departure. An event must pass all of these gates in both candidates and both held-out evaluation cohorts before it can become a confirmation lead.

## Clock and statistical units

The primary horizon is 12 future fission opportunities, not 32 molecular observations. F4 and F12 use independent domain-separated stochastic streams. Catalytic matrix is the independent higher-level unit; candidates and cohorts remain separate. Incomplete or extinct branches are retained as nonreplaced status-bearing units. Hazard and renewal outputs use the post-fission clock, while molecular-update counts are diagnostics only.

## Validation and provenance

- Repository lock: `226830e7aea9b17bde6287f41f62fb3ea8f74496`.
- Workers: `8` with one numerical-library thread each.
- New matrices/trajectories: `0/0`; new branch streams: `53760`.
- Wall time: `1855.95` seconds; GPU hours: `0`.
- Exact full regeneration reran every stochastic stream from its frozen seed identity and reproduced every scientific frame and classification.
- S01–S18, V1/V2 and S19-L01–L40 remain unchanged.

## Caveats and boundaries

This is exploratory simulation evidence. A positive result would establish only a reproducible process committor and possibly a simulation-based short-shooting coordinate. It would not identify author code, reproduce the paper, establish Phi-r as an independent precursor, prove causal control, or establish a biological claim. A negative ordering result would constrain this precise two-return process, not all robustness, error correction, recovery or homeostasis.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/e01/test_s19_l41.py
python -m ruff check src/e01_onset_discovery/fission_clock_recurrence.py scripts/e01/run_s19_l41_fission_clock_repeated_recurrence.py tests/e01/test_s19_l41.py
python scripts/e01/run_s19_l41_fission_clock_repeated_recurrence.py --prepare-lock
python scripts/e01/run_s19_l41_fission_clock_repeated_recurrence.py
```
