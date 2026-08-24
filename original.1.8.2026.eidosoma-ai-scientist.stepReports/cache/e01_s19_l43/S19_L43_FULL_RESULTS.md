# S19-L43 — Continuous Homeostatic Recovery Gain

## Chief/human handoff

- **Step:** `E01-S19-L43-CONTINUOUS-HOMEOSTATIC-RECOVERY-GAIN-v1.0.0`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** `CONTINUOUS_HOMEOSTATIC_RECOVERY_NOT_SUPPORTED`, `PROCESS_OUTCOME_FAMILY_IDENTIFIABILITY_REQUIRED`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** immutable L42-and-earlier baseline; eight fixtures; exact replay of all 53,760 L41 F12/F4 paths; candidate-separated continuous-outcome reliability; paired anchor controls; 4,096 catalytic-matrix bootstraps; exact full regeneration; storage and artifact hashes.
- **Recommended next action:** `PROCESS_OUTCOME_FAMILY_IDENTIFIABILITY_AUDIT`.

## Frozen question

After a future fission jointly breaks parent–daughter inheritance and departs from the preceding daughter, does the first subsequent run of two inherited fissions move the composition back toward the online pre-break daughter by a reliable amount? The primary continuous value is `H(certifying daughter, pre-break daughter) - H(break daughter, pre-break daughter)`. No new threshold, run length, horizon, trajectory, matrix or branch stream was searched or generated.

The same physical break and resumption certification are scored against the actual pre-break daughter, its frozen species permutation, and an unrelated-matrix prefix daughter. These paired controls therefore hold the number of fissions, inherited fissions, resumption order, opportunities, mass and phase fixed.

## Anchor results

### F12 primary continuous-gain reliability

| evaluationCohort   | candidateId       |   eligibleStates |   meanRecoveryGain |   meanRecoveryGainLower95 |   meanRecoveryGainUpper95 |   correctedBetweenStateVariance |   correctedVarianceLower95 |   splitHalfSpearman |   splitHalfLower95 | reliabilityGatePassed   |
|:-------------------|:------------------|-----------------:|-------------------:|--------------------------:|--------------------------:|--------------------------------:|---------------------------:|--------------------:|-------------------:|:------------------------|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 |               47 |          -0.256294 |                 -0.277141 |                 -0.235576 |                      0.00490061 |                 0.0030815  |            0.896855 |           0.78906  | False                   |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 |               43 |          -0.260835 |                 -0.280542 |                 -0.239622 |                      0.00449566 |                 0.00262905 |            0.919662 |           0.834283 | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-02 |               41 |          -0.261134 |                 -0.277798 |                 -0.244436 |                      0.00268843 |                 0.00165948 |            0.784321 |           0.61539  | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 |               45 |          -0.261874 |                 -0.281996 |                 -0.241578 |                      0.00434455 |                 0.00252873 |            0.78722  |           0.624872 | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 |               35 |          -0.261943 |                 -0.283296 |                 -0.241352 |                      0.00361899 |                 0.00174919 |            0.868627 |           0.734956 | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 |               36 |          -0.258129 |                 -0.284529 |                 -0.232876 |                      0.00579862 |                 0.00317205 |            0.819305 |           0.614887 | False                   |

### Paired anchor controls

| evaluationCohort   | candidateId       | branchFamily   | comparisonId                   |   definedPairs |   meanDifference |   lower95 |   upper95 | gatePassed   |
|:-------------------|:------------------|:---------------|:-------------------------------|---------------:|-----------------:|----------:|----------:|:-------------|
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12            | PRIMARY_MINUS_SPECIES_PERMUTED |             35 |        -0.231361 | -0.252589 | -0.209345 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F12            | PRIMARY_MINUS_UNRELATED        |             35 |        -0.238678 | -0.260754 | -0.217414 | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12            | PRIMARY_MINUS_SPECIES_PERMUTED |             47 |        -0.233201 | -0.25356  | -0.211677 | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F12            | PRIMARY_MINUS_UNRELATED        |             47 |        -0.23091  | -0.249577 | -0.211792 | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12            | PRIMARY_MINUS_SPECIES_PERMUTED |             43 |        -0.234865 | -0.25498  | -0.213808 | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F12            | PRIMARY_MINUS_UNRELATED        |             43 |        -0.242405 | -0.265709 | -0.220106 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12            | PRIMARY_MINUS_SPECIES_PERMUTED |             41 |        -0.239301 | -0.255523 | -0.222189 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F12            | PRIMARY_MINUS_UNRELATED        |             41 |        -0.242882 | -0.262062 | -0.225596 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12            | PRIMARY_MINUS_SPECIES_PERMUTED |             36 |        -0.228534 | -0.25306  | -0.204036 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F12            | PRIMARY_MINUS_UNRELATED        |             36 |        -0.239548 | -0.268993 | -0.212155 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12            | PRIMARY_MINUS_SPECIES_PERMUTED |             45 |        -0.234835 | -0.253333 | -0.216545 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F12            | PRIMARY_MINUS_UNRELATED        |             45 |        -0.242316 | -0.265165 | -0.220292 | False        |

### Independent F4-to-F12 transfer

| evaluationCohort   | candidateId       | comparisonId        |   definedPairs |   spearman |   lower95 |   upper95 | gatePassed   |
|:-------------------|:------------------|:--------------------|---------------:|-----------:|----------:|----------:|:-------------|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | F4_GAIN_TO_F12_GAIN |             24 |   0.854783 |  0.627155 |  0.946254 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | F4_GAIN_TO_F12_GAIN |             21 |   0.885714 |  0.653201 |  0.975141 | False        |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | F4_GAIN_TO_F12_GAIN |             25 |   0.763846 |  0.502549 |  0.88975  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | F4_GAIN_TO_F12_GAIN |             15 |   0.832143 |  0.469287 |  0.963495 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | F4_GAIN_TO_F12_GAIN |             19 |   0.759649 |  0.408048 |  0.932241 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | F4_GAIN_TO_F12_GAIN |             23 |   0.805336 |  0.554699 |  0.931587 | False        |

### Scientific gates

| evaluationCohort   | candidateId       | continuousGainReliable   | speciesPermutationControlPassed   | unrelatedMatrixControlPassed   | anchorSpecificContinuousTargetPassed   | f4GainRankPassed   | shortShootingCoordinatePassed   |
|:-------------------|:------------------|:-------------------------|:----------------------------------|:-------------------------------|:---------------------------------------|:-------------------|:--------------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | False                    | False                             | False                          | False                                  | False              | False                           |
| L28_VALIDATION     | S12F-CANDIDATE-03 | False                    | False                             | False                          | False                                  | False              | False                           |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | False                    | False                             | False                          | False                                  | False              | False                           |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | False                    | False                             | False                          | False                                  | False              | False                           |

## Interpretation

This analysis separates ordinary inheritance frequency from continuous restoration toward a specific pre-break composition. A positive paired anchor effect cannot be created by different branch opportunities or inherited-fission counts because every anchor is evaluated on the identical branch and at the identical certification boundary. State-dependent reliability and F4-to-F12 transfer are separately required before the result can be treated as a shooting coordinate.

The outcome remains a simulator proxy. It does not establish an author implementation, paper replication, a static biomarker, Phi-r incremental value, intervention efficacy, causal control or a biological conclusion.

## Provenance and validation

- Repository lock: `0f047f988ae3721d06bc02f93abfd99f923d5e4d`.
- Workers: `8`; one numerical-library thread per worker; GPU hours `0`.
- New matrices/trajectories/branch streams: `0/0/0`.
- Exact reused branch streams: `53760`.
- Wall time: `33171.01` seconds.
- S01–S18, V1/V2 and S19-L01–L42 remain unchanged.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/e01/test_s19_l43.py
python -m ruff check src/e01_onset_discovery/heredity_recovery_gain.py scripts/e01/run_s19_l43_continuous_homeostatic_recovery_gain.py tests/e01/test_s19_l43.py
python scripts/e01/run_s19_l43_continuous_homeostatic_recovery_gain.py --prepare-lock
python scripts/e01/run_s19_l43_continuous_homeostatic_recovery_gain.py
```
