# S19-L47 Full Results — Functional Coherence versus Compositional Sufficiency

## Top summary

- **Research step:** `E01-S19-L47-FUNCTIONAL-COHERENCE-COMPOSITIONAL-SUFFICIENCY-v1.0.0`
- **Completion status:** complete; additive exploratory evidence
- **Artifacts written:** frozen source/input/seed locks, episode and catalytic-matrix tables, two fixed ridge models, 4,096 matrix bootstraps, 512 target permutations, pathway-stratum audit, six figures, validation and hash manifests
- **Validation:** PASS — immutable prior, frozen L46 table identities, fixtures, source/seed/scope locks, two exact full analysis passes, regeneration, storage and artifact hashes
- **Outcome classification:** `VECTOR_FUNCTIONAL_COHERENCE_NOT_AMPLIFIED_BEYOND_COMPOSITION`, `FUNCTIONAL_COHERENCE_NOT_INCREMENTAL_BEYOND_COMPOSITION_CHRONOLOGY`, `PATHWAY_HETEROGENEITY_NOT_IDENTIFIED`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Lay summary:** L46 found that newly inherited triples are locally coherent in catalytic, exchange and growth summaries. L47 asks whether that coherence contains anything beyond the compositional smoothness and timing that define the same three-in-a-row event. It makes no new simulation, target, threshold, branch or information-theory calculation.
- **Recommended next action:** `L48_STOCHASTIC_SHOOTING_NECESSITY_AND_EFFICIENCY` under the existing human-authorized sequence. S20, E02, author contact and intervention work remain inactive.

## Frozen question

Does the local functional coherence seen in L46 exceed (a) direct composition-H coherence and (b) a development-only model containing compositional geometry, post-break opportunity count, inherited-fission count and certification delay? A companion pathway audit asks whether immediate versus delayed online certification separates one consistent functional transition subtype.

This audit does not test old-regime restoration again. It treats catalytic activation, expected net exchange and growth/division summaries as deterministic reconstructed-simulator proxies, not experimentally measured functions.

## Inputs and methods

- Inputs: exactly 19,958 frozen L46 certified-episode rows from 280 state/landmark units and their catalytic-matrix identities.
- Independent unit: catalytic matrix; candidate 2 and candidate 3, validation and confirmation remain separate.
- Direct vector contrasts: catalytic/exchange ordered-coherence excess minus the registered composition-H coherence excess.
- Fixed models: `M0_COMPOSITION` and `M1_COMPOSITION_PLUS_CHRONOLOGY`, ridge alpha 1.0, fit only on `L28_DEVELOPMENT`, no held-out refit or tuning.
- Uncertainty: exactly 4,096 catalytic-matrix bootstraps per registered effect; 512 development-target matrix permutations per candidate/target.
- Pathway strata: earliest possible run-3 certification versus delayed certification; no outcome-derived regrouping.
- Compute: intentional serial analysis because vectorized matrix bootstraps made worker-process overhead larger than the measured workload; every numerical-library thread was fixed to one.

## Registered effect results

| evaluationCohort   | candidateId       | effectId                           |   meanValue |   lower95 |   upper95 |
|:-------------------|:------------------|:-----------------------------------|------------:|----------:|----------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 | M1_RESIDUAL::CATALYTIC_ACTIVATION  |    0.006403 | -0.001292 |  0.014126 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | M1_RESIDUAL::EXPECTED_NET_EXCHANGE |    0.006875 | -0.000110 |  0.014070 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | M1_RESIDUAL::GROWTH_DIVISION       |    0.044014 |  0.005906 |  0.083696 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | ACTIVATION_MINUS_COMPOSITION       |    0.001149 | -0.005894 |  0.008745 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | NET_EXCHANGE_MINUS_COMPOSITION     |   -0.003240 | -0.010374 |  0.004100 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | M1_RESIDUAL::CATALYTIC_ACTIVATION  |    0.010918 |  0.002431 |  0.019574 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | M1_RESIDUAL::EXPECTED_NET_EXCHANGE |    0.005731 | -0.002199 |  0.013485 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | M1_RESIDUAL::GROWTH_DIVISION       |   -0.013032 | -0.046561 |  0.020484 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | ACTIVATION_MINUS_COMPOSITION       |   -0.002522 | -0.011501 |  0.005721 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | NET_EXCHANGE_MINUS_COMPOSITION     |   -0.008201 | -0.016688 |  0.000195 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | M1_RESIDUAL::CATALYTIC_ACTIVATION  |   -0.009923 | -0.020010 | -0.000435 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | M1_RESIDUAL::EXPECTED_NET_EXCHANGE |   -0.006113 | -0.015307 |  0.002791 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | M1_RESIDUAL::GROWTH_DIVISION       |    0.045688 |  0.008008 |  0.088344 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | ACTIVATION_MINUS_COMPOSITION       |   -0.011755 | -0.020721 | -0.002830 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | NET_EXCHANGE_MINUS_COMPOSITION     |   -0.014371 | -0.023183 | -0.005627 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | M1_RESIDUAL::CATALYTIC_ACTIVATION  |    0.000532 | -0.010659 |  0.011728 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | M1_RESIDUAL::EXPECTED_NET_EXCHANGE |   -0.006120 | -0.016484 |  0.003944 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | M1_RESIDUAL::GROWTH_DIVISION       |   -0.011774 | -0.046457 |  0.018718 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | ACTIVATION_MINUS_COMPOSITION       |   -0.006327 | -0.015207 |  0.002512 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | NET_EXCHANGE_MINUS_COMPOSITION     |   -0.011949 | -0.021002 | -0.003078 |

## Held-out model performance

| evaluationCohort   | candidateId       | targetId              | modelId                        |     rmse |   rSquared |   residualMean |   spearman |
|:-------------------|:------------------|:----------------------|:-------------------------------|---------:|-----------:|---------------:|-----------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 | CATALYTIC_ACTIVATION  | M0_COMPOSITION                 | 0.026977 |   0.722457 |       0.005623 |   0.814694 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | CATALYTIC_ACTIVATION  | M0_COMPOSITION                 | 0.030638 |   0.589468 |      -0.004977 |   0.770732 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | CATALYTIC_ACTIVATION  | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.028403 |   0.692339 |       0.006403 |   0.807755 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | CATALYTIC_ACTIVATION  | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.032853 |   0.527949 |      -0.009923 |   0.742777 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | EXPECTED_NET_EXCHANGE | M0_COMPOSITION                 | 0.025311 |   0.709973 |       0.005996 |   0.824082 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | EXPECTED_NET_EXCHANGE | M0_COMPOSITION                 | 0.029102 |   0.563518 |      -0.001310 |   0.748593 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | EXPECTED_NET_EXCHANGE | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.026687 |   0.677574 |       0.006875 |   0.812449 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | EXPECTED_NET_EXCHANGE | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.029692 |   0.545661 |      -0.006113 |   0.740901 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | GROWTH_DIVISION       | M0_COMPOSITION                 | 0.136280 |  -0.407038 |       0.034878 |  -0.156224 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | GROWTH_DIVISION       | M0_COMPOSITION                 | 0.139960 |  -0.829004 |       0.065423 |   0.066979 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | GROWTH_DIVISION       | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.145261 |  -0.598608 |       0.044014 |   0.209286 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | GROWTH_DIVISION       | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.137589 |  -0.767560 |       0.045688 |   0.273358 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | CATALYTIC_ACTIVATION  | M0_COMPOSITION                 | 0.029721 |   0.525605 |       0.006955 |   0.653685 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | CATALYTIC_ACTIVATION  | M0_COMPOSITION                 | 0.030361 |   0.537168 |       0.002490 |   0.609193 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | CATALYTIC_ACTIVATION  | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.032427 |   0.435288 |       0.010918 |   0.675870 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | CATALYTIC_ACTIVATION  | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.035555 |   0.365235 |       0.000532 |   0.553471 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | EXPECTED_NET_EXCHANGE | M0_COMPOSITION                 | 0.028173 |   0.533076 |       0.003341 |   0.656375 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | EXPECTED_NET_EXCHANGE | M0_COMPOSITION                 | 0.030746 |   0.499989 |      -0.005155 |   0.606754 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | EXPECTED_NET_EXCHANGE | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.029229 |   0.497422 |       0.005731 |   0.650420 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | EXPECTED_NET_EXCHANGE | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.034092 |   0.385236 |      -0.006120 |   0.543340 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | GROWTH_DIVISION       | M0_COMPOSITION                 | 0.128746 |   0.056353 |      -0.006109 |   0.143433 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | GROWTH_DIVISION       | M0_COMPOSITION                 | 0.124881 |   0.132502 |      -0.017973 |   0.337523 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | GROWTH_DIVISION       | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.121506 |   0.159498 |      -0.013032 |   0.390828 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | GROWTH_DIVISION       | M1_COMPOSITION_PLUS_CHRONOLOGY | 0.105574 |   0.380005 |      -0.011774 |   0.630582 |

Negative held-out R² means that the frozen development model is worse than predicting the held-out cohort mean. A residual above zero is not automatically independent organization; it passes the registered gate only when its matrix-bootstrap lower bound exceeds zero in every candidate/cohort group.

## Immediate versus delayed certification

| evaluationCohort   | candidateId       | targetId              |   matrixCount |   immediateMatrices |   delayedMatrices |   pairedMatrices |   immediateBranches |   delayedBranches |   meanDelayedMinusImmediate |   lower95 |   upper95 | intervalExcludesZero   | direction   |
|:-------------------|:------------------|:----------------------|--------------:|--------------------:|------------------:|-----------------:|--------------------:|------------------:|----------------------------:|----------:|----------:|:-----------------------|:------------|
| L28_VALIDATION     | S12F-CANDIDATE-02 | CATALYTIC_ACTIVATION  |            49 |                  49 |                46 |               46 |                1729 |              1488 |                    0.021864 |  0.001604 |  0.044834 | True                   | POSITIVE    |
| L28_VALIDATION     | S12F-CANDIDATE-02 | EXPECTED_NET_EXCHANGE |            49 |                  49 |                46 |               46 |                1729 |              1488 |                    0.019447 | -0.000527 |  0.041906 | False                  | POSITIVE    |
| L28_VALIDATION     | S12F-CANDIDATE-02 | GROWTH_DIVISION       |            49 |                  49 |                46 |               46 |                1729 |              1488 |                    0.038184 | -0.029814 |  0.136896 | False                  | POSITIVE    |
| L28_VALIDATION     | S12F-CANDIDATE-03 | CATALYTIC_ACTIVATION  |            50 |                  50 |                48 |               48 |                2021 |              1481 |                    0.010862 | -0.007367 |  0.028842 | False                  | POSITIVE    |
| L28_VALIDATION     | S12F-CANDIDATE-03 | EXPECTED_NET_EXCHANGE |            50 |                  50 |                48 |               48 |                2021 |              1481 |                    0.010526 | -0.006702 |  0.028034 | False                  | POSITIVE    |
| L28_VALIDATION     | S12F-CANDIDATE-03 | GROWTH_DIVISION       |            50 |                  50 |                48 |               48 |                2021 |              1481 |                   -0.001772 | -0.030591 |  0.032641 | False                  | NEGATIVE    |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | CATALYTIC_ACTIVATION  |            40 |                  40 |                40 |               40 |                1696 |              1317 |                    0.035563 |  0.009032 |  0.070487 | True                   | POSITIVE    |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | EXPECTED_NET_EXCHANGE |            40 |                  40 |                40 |               40 |                1696 |              1317 |                    0.033810 |  0.008314 |  0.069343 | True                   | POSITIVE    |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | GROWTH_DIVISION       |            40 |                  40 |                40 |               40 |                1696 |              1317 |                   -0.041324 | -0.076927 | -0.008899 | True                   | NEGATIVE    |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | CATALYTIC_ACTIVATION  |            40 |                  40 |                38 |               38 |                1504 |              1346 |                    0.018920 | -0.000836 |  0.038684 | False                  | POSITIVE    |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | EXPECTED_NET_EXCHANGE |            40 |                  40 |                38 |               38 |                1504 |              1346 |                    0.019780 |  0.000293 |  0.039029 | True                   | POSITIVE    |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | GROWTH_DIVISION       |            40 |                  40 |                38 |               38 |                1504 |              1346 |                   -0.042533 | -0.087012 | -0.004756 | True                   | NEGATIVE    |

## Scientific gates

| gateId                                        |   requiredRows |   observedRows | passed   | criterion                                                                                                      |
|:----------------------------------------------|---------------:|---------------:|:---------|:---------------------------------------------------------------------------------------------------------------|
| VECTOR_DIRECT_CONTRAST_ALL_EVALUATION_GROUPS  |              8 |              8 | False    | activation-minus-composition and exchange-minus-composition lower 95% bound above zero in four held-out groups |
| M1_RESIDUAL_ALL_TARGETS_ALL_EVALUATION_GROUPS |             12 |             12 | False    | composition-plus-chronology residual lower 95% bound above zero for three targets in four held-out groups      |
| COMPLETE_INCREMENTAL_FUNCTIONAL_COHERENCE     |              2 |              2 | False    | both direct and residual contracts pass                                                                        |
| PATHWAY_STRATUM_AVAILABILITY                  |             12 |             12 | True     | at least 20 immediate, delayed and paired matrices per target and held-out group                               |
| PATHWAY_COMMON_DIRECTION                      |             12 |             12 | False    | all delayed-minus-immediate intervals exclude zero in one common direction                                     |

## Permutation controls

| evaluationCohort   | candidateId       | targetId              |   permutations |   actualRSquared |   nullMedianRSquared |   rSquaredEmpiricalP |   actualRmse |   nullMedianRmse |   rmseEmpiricalP |
|:-------------------|:------------------|:----------------------|---------------:|-----------------:|---------------------:|---------------------:|-------------:|-----------------:|-----------------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 | CATALYTIC_ACTIVATION  |            512 |         0.692339 |            -0.152746 |             0.001949 |     0.028403 |         0.054979 |         0.001949 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | EXPECTED_NET_EXCHANGE |            512 |         0.677574 |            -0.139650 |             0.001949 |     0.026687 |         0.050173 |         0.001949 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | GROWTH_DIVISION       |            512 |        -0.598608 |            -0.480778 |             0.629630 |     0.145261 |         0.139805 |         0.629630 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | CATALYTIC_ACTIVATION  |            512 |         0.435288 |            -0.177602 |             0.005848 |     0.032427 |         0.046826 |         0.005848 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | EXPECTED_NET_EXCHANGE |            512 |         0.497422 |            -0.190552 |             0.001949 |     0.029229 |         0.044987 |         0.001949 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | GROWTH_DIVISION       |            512 |         0.159498 |            -0.073768 |             0.126706 |     0.121506 |         0.137336 |         0.126706 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | CATALYTIC_ACTIVATION  |            512 |         0.527949 |            -0.171405 |             0.001949 |     0.032853 |         0.051753 |         0.001949 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | EXPECTED_NET_EXCHANGE |            512 |         0.545661 |            -0.143487 |             0.001949 |     0.029692 |         0.047104 |         0.001949 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | GROWTH_DIVISION       |            512 |        -0.767560 |            -0.875478 |             0.421053 |     0.137589 |         0.141727 |         0.421053 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | CATALYTIC_ACTIVATION  |            512 |         0.365235 |            -0.212533 |             0.017544 |     0.035555 |         0.049141 |         0.017544 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | EXPECTED_NET_EXCHANGE |            512 |         0.385236 |            -0.205615 |             0.019493 |     0.034092 |         0.047742 |         0.019493 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | GROWTH_DIVISION       |            512 |         0.380005 |            -0.068710 |             0.001949 |     0.105574 |         0.138609 |         0.001949 |

## Interpretation boundary

The source-equation functional vectors are deterministic functions of composition and the catalytic matrix. Therefore a positive unadjusted coherence value is not evidence of a distinct functional memory. Only a preregistered increment beyond direct H and chronology could support that narrower claim. Conversely, failure of this audit does not erase L44's modest ordering result or L46's descriptive local coherence; it constrains their interpretation.

No result here establishes a universal replicator label, early warning, causal emergence, intervention efficacy, causal control or author-code identity. Prior S18 and S19 classifications remain unchanged.

## Validation, runtime and provenance

- Repository lock: `f700c8de97d47d8c4ed77a2072d0e96ff5286e15`.
- Wall time: `39.986` seconds; estimated CPU upper bound: `0.011107` hours.
- Workers: `1`; numerical-library threads: `1`; GPU hours: `0`.
- New matrices/trajectories/branches: `0/0/0`.
- Exact complete analysis passes: `2`.
- Custom code: `src/e01_onset_discovery/functional_coherence_sufficiency.py` and `scripts/e01/run_s19_l47_functional_coherence_sufficiency.py`.
- Runtime libraries: NumPy, pandas, SciPy, PyArrow and Matplotlib from the existing workspace environment.

## Limitations

L47 is an adaptive exploratory audit after many prior loops. Its fixed linear controls can test the registered mean-residual claim but cannot prove that every nonlinear compositional transformation has been exhausted. Growth summaries are reconstructed-simulator diagnostics, and certification timing is measured only for branches that achieved the frozen run-3 event. The result is therefore bounded evidence about the specific L46 coherence claim, not a global impossibility theorem about functional organization.
