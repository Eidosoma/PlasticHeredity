# S19-L42 — Fission-Conditioned Heredity Recovery Hazard

## Chief/human handoff

- **Step:** `E01-S19-L42-FISSION-CONDITIONED-HEREDITY-RECOVERY-HAZARD-v1.0.0`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** `NO_RELIABLE_HOMEOSTATIC_RECOVERY_COMMITTOR_AT_F12`, `PROCESS_TARGET_REQUIRES_REDEFINITION`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** immutable L41-and-earlier baseline; seven fixtures; exact replay of all 53,760 L41 F12/F4 paths; new analysis-seed firewall; candidate-separated variable-denominator committor reliability; 4,096 matrix bootstraps; exact full regeneration; storage and artifact hashes.
- **Recommended next action:** `PROCESS_OUTCOME_FAMILY_IDENTIFIABILITY_AUDIT`.

## Frozen question

After the first future fission that both breaks strict parent-daughter inheritance (`H<=0.9`) and moves outside the strict-H neighbourhood of the preceding selected daughter, is there a reliable conditional probability of restoring a sustained hereditary regime in that same compositional neighbourhood?

The primary event requires two consecutive later fissions for which both parent-daughter inheritance and daughter-to-pre-break-anchor similarity are strict `H>0.9`. Online certification occurs only at the second qualifying recovery fission. Uninterrupted inheritance is excluded. Inheritance resumption without return to the pre-break neighbourhood is retained as a separate baseline.

## Anchor results

### Break, inheritance-resumption and same-neighbourhood recovery

| evaluationCohort   | candidateId       |   states |   eligibleStates |   meanBreakProbability |   meanConditionalRecovery |   meanConditionalResumption |   meanOrderNull |
|:-------------------|:------------------|---------:|-----------------:|-----------------------:|--------------------------:|----------------------------:|----------------:|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 |       50 |               48 |               0.737969 |                0.00412281 |                    0.89921  |      0.00415816 |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 |       50 |               47 |               0.724375 |                0.00340355 |                    0.875332 |      0.0018725  |
| L28_VALIDATION     | S12F-CANDIDATE-02 |       50 |               41 |               0.644375 |                0.00940057 |                    0.882243 |      0.00528849 |
| L28_VALIDATION     | S12F-CANDIDATE-03 |       50 |               47 |               0.690156 |                0.00592163 |                    0.890234 |      0.00425442 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |       40 |               36 |               0.717578 |                0.00368931 |                    0.900028 |      0.00234475 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |       40 |               37 |               0.734961 |                0.00238574 |                    0.875093 |      0.0088224  |

### Conditional committor reliability

| evaluationCohort   | candidateId       | branchFamily   | targetId                  |   states |   eligibleStates |   meanConditionalQ |   minimumConditionalQ |   maximumConditionalQ |   intermediateStateCount |   meanBreakTrials |   observedBetweenStateVariance |   estimatedBinomialNoiseVariance |   correctedBetweenStateVariance |   correctedVarianceLower95 |   correctedVarianceUpper95 |   splitHalfSpearman |   splitHalfLower95 |   splitHalfUpper95 | reliabilityGatePassed   |
|:-------------------|:------------------|:---------------|:--------------------------|---------:|-----------------:|-------------------:|----------------------:|----------------------:|-------------------------:|------------------:|-------------------------------:|---------------------------------:|--------------------------------:|---------------------------:|---------------------------:|--------------------:|-------------------:|-------------------:|:------------------------|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12            | PRIMARY_PREBREAK_DAUGHTER |       50 |               48 |         0.00342654 |                     0 |             0.0789474 |                        0 |           97.875  |                    0.000150595 |                      6.86696e-05 |                     8.19254e-05 |               -6.7293e-06  |                0.000246302 |           0.216488  |          -0.106253 |           0.694496 | False                   |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12            | PRIMARY_PREBREAK_DAUGHTER |       50 |               47 |         0.00269573 |                     0 |             0.030303  |                        0 |           97.4255 |                    5.12684e-05 |                      4.73652e-05 |                     3.90323e-06 |               -1.00623e-05 |                2.00932e-05 |           0.209204  |          -0.104971 |           0.667155 | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12            | PRIMARY_PREBREAK_DAUGHTER |       50 |               41 |         0.00690633 |                     0 |             0.0769231 |                        0 |           97.0244 |                    0.000310981 |                      0.000122545 |                     0.000188436 |                2.46673e-05 |                0.000367186 |           0.658232  |           0.240787 |           0.932582 | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12            | PRIMARY_PREBREAK_DAUGHTER |       50 |               47 |         0.00436537 |                     0 |             0.05      |                        0 |           93.2128 |                    0.000129922 |                      6.82866e-05 |                     6.16351e-05 |                2.69823e-06 |                0.000134929 |           0.175755  |          -0.117599 |           0.61691  | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12            | PRIMARY_PREBREAK_DAUGHTER |       40 |               36 |         0.00409923 |                     0 |             0.027027  |                        0 |           99      |                    5.66217e-05 |                      6.04975e-05 |                    -3.87577e-06 |               -2.61478e-05 |                1.35441e-05 |           0.0929668 |          -0.195577 |           0.490885 | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12            | PRIMARY_PREBREAK_DAUGHTER |       40 |               37 |         0.00257918 |                     0 |             0.0560748 |                        0 |          100.946  |                    9.73506e-05 |                      2.94702e-05 |                     6.78804e-05 |               -3.16477e-06 |                0.000198517 |           0.492213  |           0.364977 |           1        | False                   |

### Short shooting and registered controls

| evaluationCohort   | candidateId       | comparisonId                            | comparisonType   |   definedPairs |   pointEstimate |      lower95 |    upper95 | gatePassed   |
|:-------------------|:------------------|:----------------------------------------|:-----------------|---------------:|----------------:|-------------:|-----------:|:-------------|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12_INHERITANCE_FRACTION_VS_PRIMARY     | RANK             |             48 |     0.368249    |  0.124406    | 0.567347   | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_ORDER_NULL            | DIFFERENCE       |             48 |     0.000180192 | -0.000563614 | 0.00106036 | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_SPECIES_PERMUTED      | DIFFERENCE       |             48 |     0.00342654  |  0.000803804 | 0.00730187 | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_UNRELATED             | DIFFERENCE       |             48 |     0.00342654  |  0.000737864 | 0.00754759 | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12_RESUMPTION_VS_PRIMARY               | RANK             |             48 |    -0.273542    | -0.506555    | 0.0101636  | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY | RANK             |             48 |    -0.0558754   | -0.366609    | 0.2643     | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12_INHERITANCE_FRACTION_VS_PRIMARY     | RANK             |             47 |     0.447211    |  0.234144    | 0.622206   | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_ORDER_NULL            | DIFFERENCE       |             47 |     0.00132041  |  0.0002251   | 0.00272198 | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_SPECIES_PERMUTED      | DIFFERENCE       |             47 |     0.00269573  |  0.000866377 | 0.004878   | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_UNRELATED             | DIFFERENCE       |             47 |     0.00269573  |  0.000881255 | 0.00483455 | True         |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12_RESUMPTION_VS_PRIMARY               | RANK             |             47 |    -0.0509192   | -0.301291    | 0.208717   | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY | RANK             |             46 |     0.3188      | -0.0446695   | 0.581654   | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12_INHERITANCE_FRACTION_VS_PRIMARY     | RANK             |             41 |     0.451457    |  0.16664     | 0.655678   | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_ORDER_NULL            | DIFFERENCE       |             41 |     0.00311859  |  0.000992443 | 0.0056178  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_SPECIES_PERMUTED      | DIFFERENCE       |             41 |     0.00690633  |  0.00214275  | 0.0127126  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_UNRELATED             | DIFFERENCE       |             41 |     0.00690633  |  0.00217427  | 0.0126614  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12_RESUMPTION_VS_PRIMARY               | RANK             |             41 |    -0.0790616   | -0.440217    | 0.281933   | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY | RANK             |             41 |    -0.0735222   | -0.480615    | 0.332811   | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12_INHERITANCE_FRACTION_VS_PRIMARY     | RANK             |             47 |     0.298556    |  0.0503694   | 0.507202   | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_ORDER_NULL            | DIFFERENCE       |             47 |     0.00177363  |  0.000367327 | 0.00364856 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_SPECIES_PERMUTED      | DIFFERENCE       |             47 |     0.00436537  |  0.00146061  | 0.00780775 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_UNRELATED             | DIFFERENCE       |             47 |     0.00406986  |  0.00118728  | 0.0076945  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12_RESUMPTION_VS_PRIMARY               | RANK             |             47 |     0.104899    | -0.206293    | 0.394178   | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY | RANK             |             47 |    -0.0436184   | -0.351191    | 0.273426   | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12_INHERITANCE_FRACTION_VS_PRIMARY     | RANK             |             36 |     0.571307    |  0.308986    | 0.776571   | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_ORDER_NULL            | DIFFERENCE       |             36 |     0.00149395  |  0.000431682 | 0.0028297  | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_SPECIES_PERMUTED      | DIFFERENCE       |             36 |     0.00409923  |  0.00191658  | 0.006602   | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12_PRIMARY_MINUS_UNRELATED             | DIFFERENCE       |             36 |     0.00409923  |  0.00190364  | 0.00671438 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12_RESUMPTION_VS_PRIMARY               | RANK             |             36 |    -0.143694    | -0.50268     | 0.261544   | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY | RANK             |             36 |     0.289162    | -0.124764    | 0.614431   | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12_INHERITANCE_FRACTION_VS_PRIMARY     | RANK             |             37 |     0.0628873   | -0.196877    | 0.355976   | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_ORDER_NULL            | DIFFERENCE       |             37 |     0.00114956  | -0.000102742 | 0.00317453 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_SPECIES_PERMUTED      | DIFFERENCE       |             37 |     0.00257918  |  0.000232992 | 0.00629513 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12_PRIMARY_MINUS_UNRELATED             | DIFFERENCE       |             37 |     0.00257918  |  0.000232992 | 0.00632973 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12_RESUMPTION_VS_PRIMARY               | RANK             |             37 |    -0.120051    | -0.447113    | 0.250274   | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY | RANK             |             37 |     0.165344    | -0.0669996   | 0.370732   | False        |

### Locked scientific gates

| evaluationCohort   | candidateId       | primaryConditionalCommittorReliable   | speciesPermutationControlPassed   | unrelatedMatrixControlPassed   | fixedCountOrderControlPassed   | homeostaticRecoveryTargetPassed   | f4ProgressRankPassed   | shortShootingCoordinatePassed   |
|:-------------------|:------------------|:--------------------------------------|:----------------------------------|:-------------------------------|:-------------------------------|:----------------------------------|:-----------------------|:--------------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | False                                 | True                              | True                           | True                           | False                             | False                  | False                           |
| L28_VALIDATION     | S12F-CANDIDATE-03 | False                                 | True                              | True                           | True                           | False                             | False                  | False                           |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | False                                 | True                              | True                           | True                           | False                             | False                  | False                           |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | False                                 | True                              | True                           | False                          | False                             | False                  | False                           |

## Interpretation

This loop separates persistence without disturbance from recovery following disruption. The break itself is branch observable; the pre-break daughter becomes the frozen online recovery anchor at that moment. A species-permuted anchor and unrelated-matrix prefix composition test reference specificity without changing the break. The exact fixed-count order probability holds the number of qualifying recovery fissions and post-break opportunities fixed, asking whether their actual ordering contains more sustained recovery than expected from frequency alone.

Conditional recovery uses only branches with a genuine break. Every state must contribute at least 32 such F12 trials and at least 16 per branch half to be eligible. Variable-denominator binomial noise is removed from between-state variance. Incomplete and no-break paths are retained in unconditional and availability results and are never replaced.

## Provenance and validation

- Repository lock: `3e3c64745f661ff1aa37ea558178301d8435c081`.
- Workers: `8`; one numerical-library thread per worker; GPU hours `0`.
- New matrices/trajectories/branch streams: `0/0/0`.
- Exact reused branch streams: `53760`.
- Wall time: `768.93` seconds.
- S01–S18, V1/V2 and S19-L01–L41 remain unchanged.

## Boundaries

This remains exploratory simulator evidence. A positive result would identify a branch-half-reliable propensity for recovery after a genuine disruption, not author code, paper replication, a static biomarker, Phi-r incremental value, intervention efficacy, causal control or a biological conclusion. A negative result constrains this exact same-neighbourhood two-fission recovery definition, not every form of robustness, error correction or organization.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/e01/test_s19_l42.py
python -m ruff check src/e01_onset_discovery/heredity_recovery.py scripts/e01/run_s19_l42_fission_conditioned_heredity_recovery.py tests/e01/test_s19_l42.py
python scripts/e01/run_s19_l42_fission_conditioned_heredity_recovery.py --prepare-lock
python scripts/e01/run_s19_l42_fission_conditioned_heredity_recovery.py
```
