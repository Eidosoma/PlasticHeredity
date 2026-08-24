# E01/S15 — Reconstruct Association and Replicator-State Analyses

## Concise top summary

| Field | Result |
| --- | --- |
| Research step ID | `S15` (`E01-S15-ASSOCIATION-REPLICATOR-STATE-ANALYSES-v1.0.0`) |
| Completion status | **Complete** — only S15 was executed; S16 was not started |
| Artifacts written | 30 required paths under `/artifacts/research_steps/S15`, including all runwise, paper-like, dependence-aware, boundary, figure, validation, provenance, status, and report artifacts |
| Validation result | **PASS: 26/26 checks** |
| Outcome classification | **supportive — `LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE`**; also retain `RETROSPECTIVE_TEMPORAL_FITTING_DEPENDENCE` |
| Caveats or blockers | The target is exactly thresholded H; completed fits use the future suffix; the paper does not specify the Mann–Whitney scope or one-/two-sided semantics; past-only values are sparse post-fission endpoints; pooling is secondary only. |
| Lay summary | Both mandatory estimands pass the locked candidate-specific dependence-aware gates. Any resemblance is label-coupled and retrospective: exact H determines every primary binary target, while independently past-only fitted values and the historical label point differently. |
| Recommended next action | Hand control back. Keep S16 queued and inactive until separately started; carry forward both named estimands and all target/fitting boundaries. |

## Lay summary

This step reconstructed the paper's Figure 3 and Figure 4 association analyses from the exact frozen S13Y values, without simulating anything or refitting PhiRL. The paper is internally inconsistent about whether Figure 3 uses the emergence level or its change, so both were calculated and adjudicated separately. Both mandatory estimands pass the locked candidate-specific dependence-aware gates. The agreement is directional rather than numerically exact: reconstructed mean Spearman values are about 0.061–0.075 versus the paper's 0.139, and higher-replicator-mean counts are 74–86 versus the paper's 57.

The most important limit is mathematical rather than statistical: the primary label is exactly `Y = I(H>0.9)`. Across all 180,435 completed-fit rows and all 20,000 status-bearing prefix rows, the mismatch count was zero. Exact H therefore classifies the target perfectly, `H(Y|H)=0`, and an emergence statistic cannot add unrestricted information about that same binary target once exact H is known. Completed-fit values also use partitions and Gaussian parameters learned from the finished run. S15 can therefore support only retrospective paper resemblance, never early warning, prediction, or causal control.

## Frozen question

Do the frozen S13Y branch's Figure 3/4 associations reproduce separately for `LEVEL_ANALYSIS` and `CHANGE_ANALYSIS`, for both simulator candidates, and survive dependence-aware controls without being misread as prediction?

## Inputs and provenance

- Frozen S13Y completed values: `/artifacts/research_steps/S13Y/full_source_values.parquet` (180,435 rows).
- Frozen S13Y past-only prefix values: `/artifacts/research_steps/S13Y/prefix_endpoint_values.parquet` (20,000 status-bearing; 13,705 eligible rows).
- Exact candidates: candidate 2 (`h=0.6031526490073492`, first daughter) and candidate 3 (`h=0.5613315384859516`, random nonempty daughter), both with trimmed new entrants and the selected-daughter-boundary molecular clock.
- Exact information branch: pinned PhiRL regularized source implementation; source emergence = synergy + downward causation; additive-0.5 closure and dropped-component CLR; source-confirmed Fiedler/local PhiID semantics.
- Original arXiv v1 PDF SHA-256: `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Repository method-lock commit: `24199d400bdab07e0b959bd46aa033125f4abc6e` on `eidosoma/groups/42`; remote identity recorded in `provenance_manifest.json`.

All S13Y artifact-manifest members, all 200 trajectory-cache hashes, and all S14 artifact-manifest members were checked before and after execution. Their payloads were not modified.

## Detailed methods

### Mandatory level and change estimands

`LEVEL_ANALYSIS` uses `E_t` with same-state `Y_t`. `CHANGE_ANALYSIS` uses `E_t-E_(t-1)` with current-state `Y_t` and drops the first observation per trajectory. On completed trajectories, adjacent observations are consecutive selected molecular states. On the prefix comparator, differences span consecutive *eligible* post-fission prefix endpoints and can therefore cover unequal molecular intervals. Neither estimand was allowed to replace the other.

### Runwise correlation and paper-like inference

Every run received a two-sided Spearman correlation (primary) and two-sided Pearson correlation (secondary). A run is defined only with at least three finite values, nonconstant emergence values, and both label states. Counts retain positive, negative, zero, undefined, positive-significant, negative-significant, and nonsignificant runs at unadjusted `alpha=0.05`. Arithmetic means and medians are both reported. The paper-like one-sample diagnostic is the two-sided one-sample t-test of runwise coefficients against zero; greater-direction t, Wilcoxon, and sign-binomial tests are fixed secondary diagnostics.

### Replicator-versus-drift comparisons

Within every eligible run, S15 records replicator and drift means and medians, their differences, and asymptotic tie-corrected Mann–Whitney tests in greater and two-sided forms. Because the paper leaves its Mann–Whitney scope ambiguous, both point-pooled and unpaired run-summary versions are retained; neither is selected. Fisher combines *all* eligible within-run Mann–Whitney p-values, with ineligible runs excluded explicitly. These are paper-like diagnostics and do not solve within-run serial dependence.

### Dependence-aware controls

The stronger controls use 4,096 locked PCG64DXSM replicates. The trajectory bootstrap resamples complete trajectories within each candidate; the pooled secondary view resamples shared matrix-index clusters. Circular shifts independently rotate each complete binary sequence by a nonzero offset, preserving prevalence and cyclic episode durations, before recomputing median Spearman, Pearson, and replicator-minus-drift mean differences. All seeds derive from the frozen 256-bit root and domain identities. Candidate-specific results are primary; pooling is secondary only.

### Frozen comparators and stability boundary

The same analyses are repeated for the frozen historical post-fission label and independently fitted past-only prefix endpoints. Completed-fit emergence is also correlated with exact incoming H and negative Euclidean L2 composition change. Those stability correlations are descriptive coupling only. They cannot be treated as incremental information beyond exact H.

## Commands

```bash
PYTHONPATH=src pytest -q tests/e01/test_s15_association_replicator_state.py
PYTHONPATH=src ruff check src/e01_association_replicator_state scripts/e01/run_s15_association_replicator_state.py tests/e01/test_s15_association_replicator_state.py
PYTHONPATH=src python -m compileall -q src/e01_association_replicator_state scripts/e01/run_s15_association_replicator_state.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s15_association_replicator_state.py --output-root /artifacts/research_steps/S15
```

No simulator, source fitter, GPU process, network fetch, or package installer was invoked. CPU float64 is authoritative. Execution is serial with numerical-library thread counts fixed to one; vectorized resampling makes extra workers unnecessary.

The first canonical invocation stopped at YAML parsing before any input was loaded because one outcome-selection flag was indented beneath a sequence. A syntax-only indentation repair plus an explicit config-parse test was committed and pushed as `a24c9ff8d21b5049f467d1bf41003774ec63822d`; no S13Y value or scientific outcome was accessed before that repair. The recovered event is retained in `failure_ledger.csv`.

## Results

### Candidate-specific gate results

| analysisId      | candidateScope    |   positiveSpearmanCount |   positiveSignificantSpearmanCount |   meanSpearman |   medianSpearman |   oneSampleTTwoSidedP |   circularShiftSpearmanPositiveP |   higherReplicatorMeanCount |   medianMeanDifference |   circularShiftMeanDifferencePositiveP | associationGatePassed   | stateGatePassed   | candidateAnalysisResemblancePassed   |
|:----------------|:------------------|------------------------:|-----------------------------------:|---------------:|-----------------:|----------------------:|---------------------------------:|----------------------------:|-----------------------:|---------------------------------------:|:------------------------|:------------------|:-------------------------------------|
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 |                      78 |                                 45 |      0.0645948 |        0.0580069 |           1.15868e-10 |                      0.000244081 |                          75 |               0.395836 |                            0.000244081 | True                    | True              | True                                 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 |                      76 |                                 44 |      0.0610777 |        0.0554426 |           2.31644e-10 |                      0.000244081 |                          74 |               0.516156 |                            0.000244081 | True                    | True              | True                                 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 |                      75 |                                 55 |      0.0747127 |        0.0808634 |           4.20506e-12 |                      0.000244081 |                          86 |               0.74029  |                            0.000244081 | True                    | True              | True                                 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 |                      81 |                                 50 |      0.074908  |        0.0725941 |           2.28858e-13 |                      0.000244081 |                          84 |               0.840896 |                            0.000244081 | True                    | True              | True                                 |

The paper-like ordinary p-values and Fisher combinations must be read beside the bootstrap and circular-shift results. A nominally tiny point-level or Fisher p-value is not allowed to rescue a failed trajectory-aware gate.

### Spearman and secondary Pearson runwise summaries

| analysisId      | candidateScope    |   spearmanDefinedCount |   spearmanPositiveCount |   spearmanNegativeCount |   spearmanPositiveSignificantCount |   spearmanNegativeSignificantCount |   spearmanNonsignificantCount |   spearmanMean |   spearmanMedian |   pearsonDefinedCount |   pearsonPositiveCount |   pearsonNegativeCount |   pearsonPositiveSignificantCount |   pearsonNegativeSignificantCount |   pearsonNonsignificantCount |   pearsonMean |   pearsonMedian |
|:----------------|:------------------|-----------------------:|------------------------:|------------------------:|-----------------------------------:|-----------------------------------:|------------------------------:|---------------:|-----------------:|----------------------:|-----------------------:|-----------------------:|----------------------------------:|----------------------------------:|-----------------------------:|--------------:|----------------:|
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 |                     99 |                      75 |                      24 |                                 55 |                                  4 |                            40 |      0.0747127 |        0.0808634 |                    99 |                     86 |                     13 |                                64 |                                 2 |                           33 |     0.118083  |        0.120229 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 |                     96 |                      81 |                      15 |                                 50 |                                  4 |                            42 |      0.074908  |        0.0725941 |                    96 |                     84 |                     12 |                                62 |                                 4 |                           30 |     0.12362   |        0.102207 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 |                     99 |                      78 |                      21 |                                 45 |                                  8 |                            46 |      0.0645948 |        0.0580069 |                    99 |                     75 |                     24 |                                56 |                                 9 |                           34 |     0.0920981 |        0.081273 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 |                     99 |                      76 |                      23 |                                 44 |                                  4 |                            51 |      0.0610777 |        0.0554426 |                    99 |                     74 |                     25 |                                53 |                                 9 |                           37 |     0.100815  |        0.091973 |

All significance counts use ordinary unadjusted two-sided within-run p-values and therefore remain paper-like diagnostics.

### Paper-like one-sample diagnostics

| analysisId      | candidateScope    | correlationMeasure   |   definedCount |      mean |    median |   oneSampleT |   oneSampleTTwoSidedP |   oneSampleTGreaterP |   wilcoxonGreaterP |   positiveSignCount |   nonzeroSignCount |   binomialSignGreaterP |
|:----------------|:------------------|:---------------------|---------------:|----------:|----------:|-------------:|----------------------:|---------------------:|-------------------:|--------------------:|-------------------:|-----------------------:|
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | PEARSON              |             99 | 0.118083  | 0.120229  |      9.74679 |           4.29078e-16 |          2.14539e-16 |        5.84029e-14 |                  86 |                 99 |            1.14449e-14 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | SPEARMAN             |             99 | 0.0747127 | 0.0808634 |      7.89699 |           4.20506e-12 |          2.10253e-12 |        1.24264e-10 |                  75 |                 99 |            1.38329e-07 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | PEARSON              |             96 | 0.12362   | 0.102207  |      9.35797 |           3.89528e-15 |          1.94764e-15 |        1.72272e-13 |                  84 |                 96 |            9.1581e-15  |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | SPEARMAN             |             96 | 0.074908  | 0.0725941 |      8.52866 |           2.28858e-13 |          1.14429e-13 |        1.25509e-11 |                  81 |                 96 |            2.01326e-12 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | PEARSON              |             99 | 0.0920981 | 0.081273  |      6.96172 |           3.85499e-10 |          1.92749e-10 |        8.13629e-10 |                  75 |                 99 |            1.38329e-07 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | SPEARMAN             |             99 | 0.0645948 | 0.0580069 |      7.21353 |           1.15868e-10 |          5.7934e-11  |        1.54718e-09 |                  78 |                 99 |            3.44116e-09 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | PEARSON              |             99 | 0.100815  | 0.091973  |      7.25139 |           9.66151e-11 |          4.83075e-11 |        8.4953e-10  |                  74 |                 99 |            4.253e-07   |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | SPEARMAN             |             99 | 0.0610777 | 0.0554426 |      7.06873 |           2.31644e-10 |          1.15822e-10 |        2.46192e-09 |                  76 |                 99 |            4.26716e-08 |

The two-sided one-sample t-test is the paper-matched diagnostic. The greater-direction t-test, Wilcoxon signed-rank test, and positive-sign binomial test were frozen secondary checks, not substitutes chosen after outcome access.

### Runwise replicator-versus-drift summaries

| analysisId      | candidateScope    |   definedStateComparisonCount |   higherReplicatorMeanCount |   higherReplicatorMedianCount |   acrossRunMedianDriftMean |   acrossRunMedianReplicatorMean |   medianMeanDifference |   medianMedianDifference |   positiveSignificantWithinRunMannWhitneyCount |
|:----------------|:------------------|------------------------------:|----------------------------:|------------------------------:|---------------------------:|--------------------------------:|-----------------------:|-------------------------:|-----------------------------------------------:|
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 |                            99 |                          86 |                            75 |                  -0.722893 |                      0.00933332 |               0.74029  |                 0.388046 |                                             59 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 |                            96 |                          84 |                            79 |                  -0.825722 |                      0.0103496  |               0.840896 |                 0.601013 |                                             54 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 |                            99 |                          75 |                            74 |                   0.180266 |                      0.603451   |               0.395836 |                 0.239675 |                                             52 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 |                            99 |                          74 |                            73 |                   0.152374 |                      0.657219   |               0.516156 |                 0.313741 |                                             47 |

### Fisher combinations

| analysisId      | candidateScope    |   includedRunCount |   fisherStatistic |   degreesOfFreedom |    combinedP | combinedPUnderflowedToZero   |
|:----------------|:------------------|-------------------:|------------------:|-------------------:|-------------:|:-----------------------------|
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 |                 99 |           1718.44 |                198 | 2.91965e-240 | False                        |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 |                 96 |           1653.23 |                192 | 1.53991e-230 | False                        |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 |                 99 |           1455.97 |                198 | 2.60514e-190 | False                        |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 |                 99 |           1436.47 |                198 | 1.19774e-186 | False                        |

Fisher p-values that underflow to numeric zero retain their test statistic and degrees of freedom in the machine-readable table. They remain diagnostics because each run's molecular steps are serially dependent.

### Mann–Whitney scope reconstructions

| analysisId      | candidateScope    | diagnosticScope                   |   replicatorValueCount |   driftValueCount |   mannWhitneyU |   mannWhitneyGreaterP |   mannWhitneyTwoSidedP |   rankBiserialReplicatorGreater |
|:----------------|:------------------|:----------------------------------|-----------------------:|------------------:|---------------:|----------------------:|-----------------------:|--------------------------------:|
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | POINT_POOLED_WITHIN_SCOPE         |                  85733 |              1654 |    9.89972e+07 |          1.46598e-168 |           2.93196e-168 |                        0.39627  |
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | RUN_SUMMARY_UNPAIRED_WITHIN_SCOPE |                     99 |                99 | 8582           |          3.40909e-20  |           6.81817e-20  |                        0.75125  |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | POINT_POOLED_WITHIN_SCOPE         |                  91268 |              1580 |    1.01407e+08 |          1.04887e-169 |           2.09774e-169 |                        0.406442 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | RUN_SUMMARY_UNPAIRED_WITHIN_SCOPE |                     96 |                96 | 8092           |          7.27504e-20  |           1.45501e-19  |                        0.756076 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | POINT_POOLED_WITHIN_SCOPE         |                  85787 |              1700 |    9.6071e+07  |          6.00156e-112 |           1.20031e-111 |                        0.317504 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | RUN_SUMMARY_UNPAIRED_WITHIN_SCOPE |                     99 |                99 | 7530           |          3.49017e-11  |           6.98033e-11  |                        0.536578 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | POINT_POOLED_WITHIN_SCOPE         |                  91323 |              1625 |    9.94411e+07 |          7.41331e-123 |           1.48266e-122 |                        0.340178 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | RUN_SUMMARY_UNPAIRED_WITHIN_SCOPE |                     99 |                99 | 7406           |          2.59079e-10  |           5.18157e-10  |                        0.511274 |

The point-pooled and unpaired run-summary scopes answer different questions. The paper does not identify which it used, so E01-C020 remains underdetermined at the paper-implementation level even when both fixed diagnostics point upward.

### Trajectory bootstrap and circular-shift controls

| analysisId      | candidateScope    | metric               |   observed |   bootstrapLower95 |   bootstrapUpper95 |   circularShiftNullMedian |   circularShiftNullLower95 |   circularShiftNullUpper95 |   circularShiftPositiveP |   circularShiftTwoSidedP |
|:----------------|:------------------|:---------------------|-----------:|-------------------:|-------------------:|--------------------------:|---------------------------:|---------------------------:|-------------------------:|-------------------------:|
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | medianSpearman       |  0.0580069 |          0.0388794 |          0.0786571 |               0.000207509 |                -0.00845763 |                 0.00886568 |              0.000244081 |              0.000244081 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | medianMeanDifference |  0.395836  |          0.301063  |          0.571473  |              -0.0115546   |                -0.0456955  |                 0.0242843  |              0.000244081 |              0.000244081 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | medianSpearman       |  0.0554426 |          0.0310907 |          0.0760798 |               0.000196783 |                -0.00835343 |                 0.0085187  |              0.000244081 |              0.000244081 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | medianMeanDifference |  0.516156  |          0.277178  |          0.61253   |              -0.0163775   |                -0.0523037  |                 0.022301   |              0.000244081 |              0.000244081 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | medianSpearman       |  0.0808634 |          0.0568649 |          0.0936903 |               4.18121e-05 |                -0.00859259 |                 0.0085496  |              0.000244081 |              0.000244081 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | medianMeanDifference |  0.74029   |          0.603448  |          0.905458  |              -0.00126009  |                -0.0439862  |                 0.0415728  |              0.000244081 |              0.000244081 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | medianSpearman       |  0.0725941 |          0.0474333 |          0.0920709 |              -0.000107055 |                -0.00859968 |                 0.00806109 |              0.000244081 |              0.000244081 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | medianMeanDifference |  0.840896  |          0.600809  |          1.00625   |               0.000392172 |                -0.0464352  |                 0.0454727  |              0.000244081 |              0.000244081 |

Both candidate-specific estimands have trajectory-bootstrap intervals above zero and the minimum attainable plus-one circular-shift p-value (`1/4097`) for the association and state-difference medians. This supports the locked retrospective resemblance gate but does not cure label circularity or completed-fit future dependence.

### Completed-fit, historical-label, and past-only directions

| analysisId      | candidateScope    | branchId                            |   spearmanDefinedCount |   spearmanPositiveCount |   spearmanMean |   spearmanMedian |
|:----------------|:------------------|:------------------------------------|-----------------------:|------------------------:|---------------:|-----------------:|
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | COMPLETED_FIT_HISTORICAL_COMPARATOR |                    100 |                      14 |     -0.0310559 |       -0.0287559 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | COMPLETED_FIT_HISTORICAL_COMPARATOR |                     98 |                      14 |     -0.0279372 |       -0.0298209 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | COMPLETED_FIT_HISTORICAL_COMPARATOR |                    100 |                      41 |     -0.0112898 |       -0.0112402 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | COMPLETED_FIT_HISTORICAL_COMPARATOR |                     98 |                      39 |     -0.0232553 |       -0.0208097 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | COMPLETED_FIT_PRIMARY               |                     99 |                      75 |      0.0747127 |        0.0808634 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | COMPLETED_FIT_PRIMARY               |                     96 |                      81 |      0.074908  |        0.0725941 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | COMPLETED_FIT_PRIMARY               |                     99 |                      78 |      0.0645948 |        0.0580069 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | COMPLETED_FIT_PRIMARY               |                     99 |                      76 |      0.0610777 |        0.0554426 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | PAST_ONLY_PREFIX_COMPARATOR         |                     85 |                      29 |     -0.0337868 |       -0.0313538 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | PAST_ONLY_PREFIX_COMPARATOR         |                     84 |                      33 |     -0.0177718 |       -0.0266717 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | PAST_ONLY_PREFIX_COMPARATOR         |                     85 |                      29 |     -0.0576588 |       -0.0740722 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | PAST_ONLY_PREFIX_COMPARATOR         |                     84 |                      30 |     -0.0571889 |       -0.0693196 |

The frozen historical label and independently refit past-only endpoints are evidentiary comparators, not alternatives available for favorable selection. Their differing directions establish label-scope and retrospective-fitting dependence.

### Ordinary H and composition-stability coupling

| analysisId      | candidateScope    | predictorId                                     |   definedCount |   medianCorrelation |   bootstrapMedianLower95 |   bootstrapMedianUpper95 |
|:----------------|:------------------|:------------------------------------------------|---------------:|--------------------:|-------------------------:|-------------------------:|
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | EXACT_INCOMING_H                                |            100 |            0.2017   |                 0.17369  |                 0.239554 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | EXACT_INCOMING_H                                |            100 |            0.193634 |                 0.172304 |                 0.215365 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-02 | NEGATIVE_EUCLIDEAN_L2_CLOSED_COMPOSITION_CHANGE |            100 |            0.281091 |                 0.241517 |                 0.293325 |
| CHANGE_ANALYSIS | S12F-CANDIDATE-03 | NEGATIVE_EUCLIDEAN_L2_CLOSED_COMPOSITION_CHANGE |            100 |            0.25779  |                 0.222906 |                 0.27748  |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | EXACT_INCOMING_H                                |            100 |            0.269943 |                 0.226782 |                 0.304879 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | EXACT_INCOMING_H                                |            100 |            0.270713 |                 0.223176 |                 0.292696 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-02 | NEGATIVE_EUCLIDEAN_L2_CLOSED_COMPOSITION_CHANGE |            100 |            0.277763 |                 0.231784 |                 0.319605 |
| LEVEL_ANALYSIS  | S12F-CANDIDATE-03 | NEGATIVE_EUCLIDEAN_L2_CLOSED_COMPOSITION_CHANGE |            100 |            0.275472 |                 0.237035 |                 0.302918 |

These associations make the reaction-coordinate boundary explicit. They do not change the exact identity `Y=I(H>0.9)` or create an incremental-information claim.

### Paper-target reconstruction rows

| analysisId      | status                      |   rowCount |
|:----------------|:----------------------------|-----------:|
| CHANGE_ANALYSIS | CLOSELY_RECONSTRUCTED       |          4 |
| CHANGE_ANALYSIS | DIFFERENT                   |          1 |
| CHANGE_ANALYSIS | DIRECTIONALLY_SIMILAR       |          7 |
| CHANGE_ANALYSIS | UNDERDETERMINED_PAPER_SCOPE |          4 |
| LEVEL_ANALYSIS  | CLOSELY_RECONSTRUCTED       |          4 |
| LEVEL_ANALYSIS  | DIFFERENT                   |          2 |
| LEVEL_ANALYSIS  | DIRECTIONALLY_SIMILAR       |          6 |
| LEVEL_ANALYSIS  | UNDERDETERMINED_PAPER_SCOPE |          4 |

`paper_target_comparison.csv` contains candidate-specific rows for E01-C015 through E01-C021 under both mandatory estimands. E01-C020 remains `UNDERDETERMINED_PAPER_SCOPE` at the claim level because the paper does not state whether its Mann–Whitney test pooled molecular steps or run summaries; both scope-specific results are preserved.

### Interpretation boundaries

| boundaryId                         | status                |   rowCount |
|:-----------------------------------|:----------------------|-----------:|
| COMPLETED_FIT_FUTURE_DEPENDENCE    | RETROSPECTIVE_ONLY    |          1 |
| EXACT_H_DETERMINISM                | STRUCTURAL_CONSTRAINT |          1 |
| HISTORICAL_POST_FISSION_DIRECTION  | DIFFERENT_DIRECTION   |          4 |
| NO_PREDICTION_OR_CAUSAL_CONTROL    | OUT_OF_SCOPE          |          1 |
| NO_UNRESTRICTED_INCREMENT_BEYOND_H | STRUCTURAL_CONSTRAINT |          1 |
| ORDINARY_STABILITY_COUPLING        | DESCRIPTIVE_COUPLING  |          8 |
| PAST_ONLY_DIRECTION                | DIRECTION_REVERSAL    |          4 |

The current molecular label, historical post-fission label, and past-only refit answer different questions. No positive completed-fit association can override the exact-H circularity, historical-label difference, past-only direction, or future-fitting boundary.

## Figures and machine-readable artifacts

- `figures/figure3_association_reconstruction.png`: candidate-specific runwise Spearman distributions for level and change.
- `figures/figure4_state_reconstruction.png`: candidate-specific runwise drift/replicator means and median ± across-run SD.
- `figures/dependence_aware_controls.png`: trajectory-bootstrap intervals versus circular-shift nulls.
- `figures/interpretation_boundaries.png`: primary, historical, past-only, exact-H, and ordinary-stability directions.
- Parquet files retain every runwise correlation/state result and all 4,096-replicate bootstrap/shift distributions. CSV files retain compact summaries, decisions, paper targets, and interpretation boundaries.

## Validation

PASS: 26/26 checks. Deterministic independent executions produced identical hashes for every derived frame, including both 73,728-row resampling distributions. Runwise anchor results exactly replayed frozen S13Y, selected statistics were independently recomputed, observed circular-shift metrics matched the runwise tables, candidate/pool roles and level/change identities were exact, and every frozen upstream hash matched before and after execution. Five focused repository tests, Ruff, and compilation passed before the canonical run. Four PNGs passed render/cardinality/dimension checks and were separately inspected; the initial Figure 3 title/legend collision was corrected before finalization.

## Dependencies and parameters

- Python `3.13.14`; NumPy `2.4.6`; pandas `2.3.3`; SciPy `1.18.0`; PyArrow `24.0.0`; Matplotlib `3.11.1`.
- 4,096 trajectory bootstraps and 4,096 nonzero circular-shift replicates per branch × analysis × candidate scope.
- CPU float64, one process, one numerical-library thread, no GPU.

## Caveats, blockers, failed assumptions, and limitations

- The binary target is exactly determined by H. This is a structural circularity constraint, not a low-power result.
- Completed-fit partition and Gaussian parameters depend on the final trajectory suffix. S15 is retrospective-only.
- The Results text names emergence levels while Figure 3 says changes in emergence. The discrepancy is preserved, and its two outcomes are not collapsed.
- The historical post-fission label is not the same target as molecular same-state `Y`. Its result cannot be discarded.
- Past-only values start only after 256 transitions and exist at eligible post-fission endpoints; their first differences span irregular molecular intervals.
- The paper does not state the Mann–Whitney scope, sidedness, tie method, or ineligible-run policy. Every fixed interpretation is labeled diagnostic.
- Point-pooled Mann–Whitney and Fisher combinations do not remove molecular-time dependence. Bootstrap and circular shifts are the stronger controls.
- Candidate pooling is secondary only and cannot rescue a candidate-specific failure.
- Public PhiRL source equivalence does not establish identity with unavailable author code.
- S15 fits no predictor and executes no intervention. It supplies no prediction, early-warning, or causal-control evidence.

## Artifact provenance

`input_manifest.json` records every frozen input and before/after SHA-256. `method_lock.json` records the pushed repository lock. `provenance_manifest.json` records code, runtime, numeric policy, command, and repository identities. `artifact_manifest.json` hashes every required output except itself. Repository source remains in Git and was not copied into artifacts.

## Recommended next action

Return control to the Chief Scientist workflow. Keep S16 queued and inactive until a separate instruction. If S16 is later started, carry forward exact-H determinism, completed-fit future dependence, both named level/change outcomes, ordinary-stability coupling, the historical-label result, and the past-only result. Do not reinterpret S15 as prediction or causal control.
