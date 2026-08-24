# S19-L40 — Online Recurrence After a Certified Departure

## Chief/human handoff

- **Step:** `E01-S19-L40-ONLINE-RECURRENCE-AFTER-DEPARTURE-COMMITTOR-v1.0.0`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** `RECURRENCE_AFTER_DEPARTURE_ORDER_NOT_SUPPORTED`, `MEMBERSHIP_FREQUENCY_SUFFICIENT`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** immutable-prior and anchor replay; exact numerical/discrete/path replay of all 53,760 frozen H32/H8 streams; seven fixtures; candidate-separated split-half reliability; three fixed anchor/order controls; 4,096 catalytic-matrix bootstraps; independent full regeneration; runtime/storage/artifact hashes.
- **Recommended next action:** `REPEATED_CROSS_GENERATION_RECURRENCE_COMMITTOR`.

## Frozen question

Does the simulator have a reliable state-dependent probability of leaving and then returning to a compositional neighborhood fixed entirely from the observed past? The sole primary anchor is the latest selected post-fission composition before the restored state. Departure is the first future post-fission boundary with `H<=0.9`; online certification is the first later future boundary with strict `H>0.9` to that same anchor.

This cannot be satisfied by ordinary adjacent smoothness: a trajectory that remains near the anchor is never positive. No completed trajectory, completed-run centroid, future-defined basin, threshold variant, anchor search, or horizon search is used.

## Frozen anchors

| evaluationCohort   | candidateId       | targetId                       |   anchors |   meanAnchorGeneration |   meanAnchorClock |
|:-------------------|:------------------|:-------------------------------|----------:|-----------------------:|------------------:|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | PREFIX_ANCHOR                  |        50 |                 12.24  |           121.38  |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | SPECIES_PERMUTED_PREFIX_ANCHOR |        50 |                 12.24  |           121.38  |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | UNRELATED_MATRIX_PREFIX_ANCHOR |        50 |                 12.24  |           121.38  |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | PREFIX_ANCHOR                  |        50 |                 12.38  |           121.44  |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | SPECIES_PERMUTED_PREFIX_ANCHOR |        50 |                 12.38  |           121.44  |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | UNRELATED_MATRIX_PREFIX_ANCHOR |        50 |                 12.38  |           121.44  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PREFIX_ANCHOR                  |        50 |                 13.34  |           122.78  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | SPECIES_PERMUTED_PREFIX_ANCHOR |        50 |                 13.34  |           122.78  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | UNRELATED_MATRIX_PREFIX_ANCHOR |        50 |                 13.34  |           122.78  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PREFIX_ANCHOR                  |        50 |                 12.84  |           122.44  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | SPECIES_PERMUTED_PREFIX_ANCHOR |        50 |                 12.84  |           122.44  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | UNRELATED_MATRIX_PREFIX_ANCHOR |        50 |                 12.84  |           122.44  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PREFIX_ANCHOR                  |        40 |                 13     |           122     |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | SPECIES_PERMUTED_PREFIX_ANCHOR |        40 |                 13     |           122     |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | UNRELATED_MATRIX_PREFIX_ANCHOR |        40 |                 13     |           122     |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PREFIX_ANCHOR                  |        40 |                 11.475 |           120.225 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | SPECIES_PERMUTED_PREFIX_ANCHOR |        40 |                 11.475 |           120.225 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | UNRELATED_MATRIX_PREFIX_ANCHOR |        40 |                 11.475 |           120.225 |

## Process probability and opportunity

| evaluationCohort   | candidateId       | branchFamily   | targetId                       |      meanQ |   meanDeparture |   meanMixedOpportunity |   meanReturnProgress |   meanOrderNull |   meanOpportunityWithoutCertification |
|:-------------------|:------------------|:---------------|:-------------------------------|-----------:|----------------:|-----------------------:|---------------------:|----------------:|--------------------------------------:|
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | H32            | PREFIX_ANCHOR                  | 0.013125   |        0.990938 |             0.126094   |           0.613054   |     0.092564    |                            0.112969   |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        1        |             0          |           0.132598   |     0           |                            0          |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        1        |             0          |           0.13899    |     0           |                            0          |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | H8             | PREFIX_ANCHOR                  | 0.000625   |        0.613437 |             0.014375   |           0.018218   |     0.0071875   |                            0.01375    |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        0.72     |             0          |           0.00508057 |     0           |                            0          |
| L28_DEVELOPMENT    | S12F-CANDIDATE-02 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        0.72     |             0          |           0.00520442 |     0           |                            0          |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | H32            | PREFIX_ANCHOR                  | 0.0170313  |        0.981563 |             0.1425     |           0.591921   |     0.0991406   |                            0.125469   |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR | 0.00078125 |        1        |             0.00078125 |           0.123987   |     0.000580729 |                            0          |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        1        |             0          |           0.137488   |     0           |                            0          |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | H8             | PREFIX_ANCHOR                  | 0.001875   |        0.654375 |             0.0125     |           0.0311244  |     0.00630208  |                            0.010625   |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        0.762813 |             0          |           0.00479558 |     0           |                            0          |
| L28_DEVELOPMENT    | S12F-CANDIDATE-03 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        0.762813 |             0          |           0.00642493 |     0           |                            0          |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            | PREFIX_ANCHOR                  | 0.0309375  |        0.973594 |             0.15625    |           0.601449   |     0.117404    |                            0.125312   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR | 0.0003125  |        1        |             0.0003125  |           0.140432   |     0.000242187 |                            0          |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR | 0.00015625 |        1        |             0.00015625 |           0.136954   |     0.000130208 |                            0          |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H8             | PREFIX_ANCHOR                  | 0.0028125  |        0.631563 |             0.0178125  |           0.0439616  |     0.00890625  |                            0.015      |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        0.767813 |             0          |           0.00950719 |     0           |                            0          |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        0.767813 |             0          |           0.00843988 |     0           |                            0          |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            | PREFIX_ANCHOR                  | 0.0153125  |        0.985469 |             0.182031   |           0.593472   |     0.127052    |                            0.166719   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        1        |             0          |           0.137431   |     0           |                            0          |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        1        |             0          |           0.107829   |     0           |                            0          |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H8             | PREFIX_ANCHOR                  | 0.0009375  |        0.558125 |             0.0103125  |           0.0322358  |     0.00526042  |                            0.009375   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        0.706562 |             0          |           0.00733477 |     0           |                            0          |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        0.706562 |             0          |           0.00495721 |     0           |                            0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            | PREFIX_ANCHOR                  | 0.0105469  |        0.987891 |             0.104102   |           0.577339   |     0.0740717   |                            0.0935547  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        1        |             0          |           0.140515   |     0           |                            0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        1        |             0          |           0.130152   |     0           |                            0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H8             | PREFIX_ANCHOR                  | 0.00078125 |        0.651172 |             0.00507813 |           0.0195503  |     0.00253906  |                            0.00429688 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        0.742578 |             0          |           0.0049763  |     0           |                            0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        0.742578 |             0          |           0.00616662 |     0           |                            0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            | PREFIX_ANCHOR                  | 0.0142578  |        0.995508 |             0.170117   |           0.602852   |     0.127002    |                            0.155859   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        1        |             0          |           0.118038   |     0           |                            0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        1        |             0          |           0.119702   |     0           |                            0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H8             | PREFIX_ANCHOR                  | 0.0015625  |        0.712891 |             0.0277344  |           0.0209945  |     0.0159505   |                            0.0261719  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR | 0          |        0.8375   |             0          |           0.00295591 |     0           |                            0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR | 0          |        0.8375   |             0          |           0.00855272 |     0           |                            0          |

## Committor reliability

| evaluationCohort   | candidateId       | branchFamily   | targetId                       |   states |   eligibleStates |      meanQ |   minimumQ |   maximumQ |   intermediateStateCount |   observedBetweenStateVariance |   estimatedBinomialNoiseVariance |   correctedBetweenStateVariance |   correctedVarianceLower95 |   correctedVarianceUpper95 |   splitHalfSpearman |   splitHalfLower95 |   splitHalfUpper95 | reliabilityGatePassed   |
|:-------------------|:------------------|:---------------|:-------------------------------|---------:|-----------------:|-----------:|-----------:|-----------:|-------------------------:|-------------------------------:|---------------------------------:|--------------------------------:|---------------------------:|---------------------------:|--------------------:|-------------------:|-------------------:|:------------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            | PREFIX_ANCHOR                  |       50 |               50 | 0.0309375  |          0 |  0.539062  |                        5 |                    0.00787219  |                      0.00017532  |                     0.00769687  |                0.00066664  |                0.0184449   |            0.598137 |          0.261565  |           0.820007 | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR |       50 |               50 | 0.0003125  |          0 |  0.015625  |                        0 |                    4.88281e-06 |                      2.42218e-06 |                     2.46063e-06 |                0           |                6.78399e-06 |            1        |          1         |           1        | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR |       50 |               50 | 0.00015625 |          0 |  0.0078125 |                        0 |                    1.2207e-06  |                      1.2207e-06  |                    -4.23516e-22 |               -1.49474e-07 |                0           |          nan        |        nan         |         nan        | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H8             | PREFIX_ANCHOR                  |       50 |               50 | 0.0028125  |          0 |  0.078125  |                        0 |                    0.000146385 |                      4.22402e-05 |                     0.000104145 |               -5.97895e-07 |                0.000285971 |            0.652205 |         -0.0291606 |           1        | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR |       50 |               50 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR |       50 |               50 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            | PREFIX_ANCHOR                  |       50 |               50 | 0.0153125  |          0 |  0.09375   |                        0 |                    0.000759726 |                      0.000112862 |                     0.000646864 |                0.000251442 |                0.00102061  |            0.810159 |          0.612741  |           0.933465 | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR |       50 |               50 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR |       50 |               50 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H8             | PREFIX_ANCHOR                  |       50 |               50 | 0.0009375  |          0 |  0.015625  |                        0 |                    1.40505e-05 |                      1.46484e-05 |                    -5.97895e-07 |               -4.18527e-06 |                0           |          nan        |        nan         |         nan        | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR |       50 |               50 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR |       50 |               50 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            | PREFIX_ANCHOR                  |       40 |               40 | 0.0105469  |          0 |  0.101562  |                        1 |                    0.000577643 |                      7.77357e-05 |                     0.000499907 |                6.53051e-05 |                0.000958108 |            0.8007   |          0.539199  |           0.954698 | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR |       40 |               40 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR |       40 |               40 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H8             | PREFIX_ANCHOR                  |       40 |               40 | 0.00078125 |          0 |  0.03125   |                        0 |                    2.44141e-05 |                      1.20133e-05 |                     1.24008e-05 |                0           |                3.34464e-05 |            1        |          1         |           1        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR |       40 |               40 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR |       40 |               40 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            | PREFIX_ANCHOR                  |       40 |               40 | 0.0142578  |          0 |  0.265625  |                        2 |                    0.00229394  |                      9.30546e-05 |                     0.00220089  |                4.22483e-05 |                0.00548478  |            0.69208  |          0.296352  |           0.924844 | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            | SPECIES_PERMUTED_PREFIX_ANCHOR |       40 |               40 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            | UNRELATED_MATRIX_PREFIX_ANCHOR |       40 |               40 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H8             | PREFIX_ANCHOR                  |       40 |               40 | 0.0015625  |          0 |  0.0625    |                        0 |                    9.76562e-05 |                      2.32515e-05 |                     7.44048e-05 |                0           |                0.00020819  |            1        |          1         |           1        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H8             | SPECIES_PERMUTED_PREFIX_ANCHOR |       40 |               40 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H8             | UNRELATED_MATRIX_PREFIX_ANCHOR |       40 |               40 | 0          |          0 |  0         |                        0 |                    0           |                      0           |                     0           |                0           |                0           |          nan        |        nan         |         nan        | False                   |

## H8 coordinate and controls

| evaluationCohort   | candidateId       | comparisonId                                 | comparisonType   |   definedPairs |   pointEstimate |     lower95 |    upper95 | gatePassed   |
|:-------------------|:------------------|:---------------------------------------------|:-----------------|---------------:|----------------:|------------:|-----------:|:-------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | CURRENT_MASS_VS_H32_PRIMARY                  | RANK             |             50 |       0.144599  | -0.140031   |  0.411553  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | GENERATION_PHASE_VS_H32_PRIMARY              | RANK             |             50 |      -0.125499  | -0.385827   |  0.159011  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32_PRIMARY_MINUS_ORDER_NULL                 | DIFFERENCE       |             50 |      -0.0864669 | -0.126241   | -0.0506057 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32_PRIMARY_MINUS_PERMUTED_ANCHOR            | DIFFERENCE       |             50 |       0.030625  |  0.0104688  |  0.0576562 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32_PRIMARY_MINUS_UNRELATED_ANCHOR           | DIFFERENCE       |             50 |       0.0307812 |  0.0101563  |  0.05875   | True         |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H8_EVENT_Q_VS_H32_PRIMARY                    | RANK             |             50 |       0.473489  |  0.231097   |  0.668713  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H8_RETURN_PROGRESS_VS_H32_PRIMARY            | RANK             |             50 |       0.0780411 | -0.250673   |  0.375984  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY | RANK             |             50 |       0.432368  |  0.146146   |  0.670429  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PREFIX_INHERITANCE_FRACTION_VS_H32_PRIMARY   | RANK             |             50 |       0.374596  |  0.108174   |  0.602337  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | CURRENT_MASS_VS_H32_PRIMARY                  | RANK             |             50 |       0.0503595 | -0.241499   |  0.34889   | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | GENERATION_PHASE_VS_H32_PRIMARY              | RANK             |             50 |      -0.210848  | -0.463694   |  0.0729213 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32_PRIMARY_MINUS_ORDER_NULL                 | DIFFERENCE       |             50 |      -0.111739  | -0.155774   | -0.0708502 | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32_PRIMARY_MINUS_PERMUTED_ANCHOR            | DIFFERENCE       |             50 |       0.0153125 |  0.0084375  |  0.023125  | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32_PRIMARY_MINUS_UNRELATED_ANCHOR           | DIFFERENCE       |             50 |       0.0153125 |  0.00828125 |  0.0235938 | True         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H8_EVENT_Q_VS_H32_PRIMARY                    | RANK             |             50 |       0.336543  |  0.0866509  |  0.551563  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H8_RETURN_PROGRESS_VS_H32_PRIMARY            | RANK             |             50 |       0.188695  | -0.0871892  |  0.452185  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY | RANK             |             50 |       0.36463   |  0.0910722  |  0.614788  | False        |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PREFIX_INHERITANCE_FRACTION_VS_H32_PRIMARY   | RANK             |             50 |       0.250375  | -0.0173609  |  0.497995  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | CURRENT_MASS_VS_H32_PRIMARY                  | RANK             |             40 |       0.0687678 | -0.302007   |  0.411163  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | GENERATION_PHASE_VS_H32_PRIMARY              | RANK             |             40 |       0.142072  | -0.162768   |  0.419295  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32_PRIMARY_MINUS_ORDER_NULL                 | DIFFERENCE       |             40 |      -0.0635248 | -0.108302   | -0.0265044 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32_PRIMARY_MINUS_PERMUTED_ANCHOR            | DIFFERENCE       |             40 |       0.0105469 |  0.00410156 |  0.01875   | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32_PRIMARY_MINUS_UNRELATED_ANCHOR           | DIFFERENCE       |             40 |       0.0105469 |  0.00429688 |  0.01875   | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H8_EVENT_Q_VS_H32_PRIMARY                    | RANK             |             40 |       0.299744  |  0.238621   |  0.556754  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H8_RETURN_PROGRESS_VS_H32_PRIMARY            | RANK             |             40 |      -0.0921932 | -0.388477   |  0.24281   | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY | RANK             |             40 |       0.323698  | -0.0293847  |  0.608279  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PREFIX_INHERITANCE_FRACTION_VS_H32_PRIMARY   | RANK             |             40 |       0.383679  |  0.0853896  |  0.625497  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | CURRENT_MASS_VS_H32_PRIMARY                  | RANK             |             40 |       0.338057  |  0.0594441  |  0.580005  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | GENERATION_PHASE_VS_H32_PRIMARY              | RANK             |             40 |      -0.123961  | -0.390295   |  0.168885  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32_PRIMARY_MINUS_ORDER_NULL                 | DIFFERENCE       |             40 |      -0.112744  | -0.180941   | -0.0545693 | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32_PRIMARY_MINUS_PERMUTED_ANCHOR            | DIFFERENCE       |             40 |       0.0142578 |  0.00253906 |  0.031958  | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32_PRIMARY_MINUS_UNRELATED_ANCHOR           | DIFFERENCE       |             40 |       0.0142578 |  0.00273437 |  0.0314453 | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H8_EVENT_Q_VS_H32_PRIMARY                    | RANK             |             40 |       0.356276  |  0.312817   |  0.653456  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H8_RETURN_PROGRESS_VS_H32_PRIMARY            | RANK             |             40 |       0.270293  | -0.0860741  |  0.597085  | False        |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY | RANK             |             40 |       0.623325  |  0.374724   |  0.786114  | True         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PREFIX_INHERITANCE_FRACTION_VS_H32_PRIMARY   | RANK             |             40 |       0.272666  | -0.00352245 |  0.494525  | False        |

The primary short coordinate is mean maximum post-departure H over the frozen H8 branches, with zero assigned when no departure occurs. Molecule-permuted and unrelated-matrix anchors test specificity. The exact order null fixes each branch's near/departed counts and randomizes only their ordering.

## Online return hazard

| evaluationCohort   | candidateId       | branchFamily   |   futureBoundaryOneBased |   branchesAtRisk |   certifications |   discreteHazard |   survivalWithoutCertification |   cumulativeCertificationIncidence |
|:-------------------|:------------------|:---------------|-------------------------:|-----------------:|-----------------:|-----------------:|-------------------------------:|-----------------------------------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                        1 |             6400 |                0 |       0          |                       1        |                         0          |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                        2 |             6331 |               82 |       0.0129521  |                       0.987048 |                         0.0129521  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                        3 |             5195 |               61 |       0.0117421  |                       0.975458 |                         0.0245421  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                        4 |             2890 |               31 |       0.0107266  |                       0.964994 |                         0.0350055  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                        5 |             1176 |               12 |       0.0102041  |                       0.955148 |                         0.0448524  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                        6 |              408 |                9 |       0.0220588  |                       0.934078 |                         0.0659218  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                        7 |               95 |                3 |       0.0315789  |                       0.904581 |                         0.095419   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                        8 |               25 |                0 |       0          |                       0.904581 |                         0.095419   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                        9 |               11 |                0 |       0          |                       0.904581 |                         0.095419   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                       10 |                4 |                0 |       0          |                       0.904581 |                         0.095419   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                       11 |                3 |                0 |       0          |                       0.904581 |                         0.095419   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                       12 |                3 |                0 |       0          |                       0.904581 |                         0.095419   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                       13 |                3 |                0 |       0          |                       0.904581 |                         0.095419   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                       14 |                2 |                0 |       0          |                       0.904581 |                         0.095419   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | H32            |                       15 |                1 |                0 |       0          |                       0.904581 |                         0.095419   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            |                        1 |             6400 |                0 |       0          |                       1        |                         0          |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            |                        2 |             6271 |               37 |       0.00590018 |                       0.9941   |                         0.00590018 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            |                        3 |             5027 |               44 |       0.00875274 |                       0.985399 |                         0.0146013  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            |                        4 |             2192 |               10 |       0.00456204 |                       0.980903 |                         0.0190967  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            |                        5 |              727 |                4 |       0.00550206 |                       0.975506 |                         0.0244937  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            |                        6 |              274 |                3 |       0.0109489  |                       0.964826 |                         0.0351744  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            |                        7 |               66 |                0 |       0          |                       0.964826 |                         0.0351744  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            |                        8 |               18 |                0 |       0          |                       0.964826 |                         0.0351744  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | H32            |                        9 |                2 |                0 |       0          |                       0.964826 |                         0.0351744  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                        1 |             5120 |                0 |       0          |                       1        |                         0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                        2 |             5072 |               22 |       0.00433754 |                       0.995662 |                         0.00433754 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                        3 |             4223 |               22 |       0.00520957 |                       0.990475 |                         0.00952451 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                        4 |             2257 |                6 |       0.0026584  |                       0.987842 |                         0.0121576  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                        5 |              885 |                4 |       0.00451977 |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                        6 |              262 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                        7 |               62 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                        8 |               26 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                        9 |               20 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                       10 |               18 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                       11 |               12 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                       12 |               10 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                       13 |                8 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                       14 |                4 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | H32            |                       15 |                1 |                0 |       0          |                       0.983378 |                         0.0166224  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                        1 |             5120 |                0 |       0          |                       1        |                         0          |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                        2 |             5035 |               17 |       0.00337637 |                       0.996624 |                         0.00337637 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                        3 |             3709 |               21 |       0.0056619  |                       0.990981 |                         0.00901915 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                        4 |             1600 |               17 |       0.010625   |                       0.980452 |                         0.0195483  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                        5 |              609 |                7 |       0.0114943  |                       0.969182 |                         0.0308179  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                        6 |              321 |                7 |       0.0218069  |                       0.948047 |                         0.0519527  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                        7 |              185 |                3 |       0.0162162  |                       0.932674 |                         0.0673264  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                        8 |              134 |                0 |       0          |                       0.932674 |                         0.0673264  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                        9 |              123 |                1 |       0.00813008 |                       0.925091 |                         0.0749091  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                       10 |              117 |                0 |       0          |                       0.925091 |                         0.0749091  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                       11 |              106 |                0 |       0          |                       0.925091 |                         0.0749091  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                       12 |               79 |                0 |       0          |                       0.925091 |                         0.0749091  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                       13 |               24 |                0 |       0          |                       0.925091 |                         0.0749091  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | H32            |                       14 |                2 |                0 |       0          |                       0.925091 |                         0.0749091  |

Post-fission certification is never projected onto intervening molecular observations. Molecular offsets remain named diagnostics only.

## Scientific gates

| evaluationCohort   | candidateId       | primaryH32Reliable   | h8ReturnProgressTransferPassed   | permutedAnchorControlPassed   | unrelatedAnchorControlPassed   | sequenceOrderControlPassed   | opportunityNondegeneracyPassed   | recurrenceAfterDepartureTargetPassed   | shortShootingCoordinatePassed   | l39InheritancePropensityPassed   | prefixInheritanceControlPassed   | massControlPassed   | phaseControlPassed   |
|:-------------------|:------------------|:---------------------|:---------------------------------|:------------------------------|:-------------------------------|:-----------------------------|:---------------------------------|:---------------------------------------|:--------------------------------|:---------------------------------|:---------------------------------|:--------------------|:---------------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | False                | False                            | True                          | True                           | False                        | True                             | False                                  | False                           | False                            | False                            | False               | False                |
| L28_VALIDATION     | S12F-CANDIDATE-03 | False                | False                            | True                          | True                           | False                        | True                             | False                                  | False                           | False                            | False                            | False               | False                |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | False                | False                            | True                          | True                           | False                        | False                            | False                                  | False                           | False                            | False                            | False               | False                |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | False                | False                            | True                          | True                           | False                        | True                             | False                                  | False                           | True                             | False                            | False               | False                |

## Validation and provenance

- Repository commit: `8cec7774d25e2409dec5558956cb1d5e79d7660f`.
- Workers: `8`; one numerical-library thread per worker.
- Wall time: `951.215` seconds.
- New matrices/trajectories/branch streams: `0/0/0`.
- All scientific frames and paths were independently regenerated from the lock.
- Every S01–S18 and S19-L01–L39 artifact remains immutable.

## Interpretation boundary

A positive result would establish only a simulator-defined return process and a conditional stochastic-shooting coordinate. It would not identify the paper label, an author implementation, a static observed biomarker, causal emergence, intervention efficacy, biological replication, or causal control.

## Next boundary

L40 is frozen. The standing human authorization permits `REPEATED_CROSS_GENERATION_RECURRENCE_COMMITTOR` as the next bounded loop through L55. S20, E02, author contact, interventions and report-bundle generation remain inactive.
