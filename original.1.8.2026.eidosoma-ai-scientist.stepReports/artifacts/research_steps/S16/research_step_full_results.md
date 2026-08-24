# E01/S16 — Reconstruct the First-25%-to-Final-75% Prediction Experiment

## Concise top summary

| Field | Result |
| --- | --- |
| Research step ID | `S16` (`E01-S16-FIRST-QUARTER-PREDICTION-RECONSTRUCTION-v1.0.0`) |
| Completion status | **Complete** — only S16 was executed; S17 was not started |
| Artifacts written | 34 required paths under `/artifacts/research_steps/S16` |
| Validation result | **PASS: 32/32 checks** |
| Outcome classification | Retrospective: **`NOT_SUPPORTED_WITHIN_TESTED_SCOPE`**; prospective: **`NOT_SUPPORTED_WITHIN_TESTED_SCOPE`** |
| Caveats or blockers | Exact contemporaneous H determines Y; completed-fit PhiRL is suffix-dependent; the frozen molecular target is highly imbalanced; the paper omits its tensor and MLP details; pooling is secondary only. |
| Lay summary | The locked experiment tested the paper-like future-trajectory task and a genuinely cutoff-only reconstruction. Completed-fit and cutoff-causal evidence were adjudicated separately against composition, counts, flux, dummy, and exact-H history controls. |
| Recommended next action | Hand control back. Keep S17 queued and inactive until separately instructed. |

## Lay summary

This step used the first quarter of each of the 200 already frozen S13Y trajectories to predict the remaining three quarters. Every learned feature used the same masked, original-order MLP and the same ten matrix-level splits. The paper-like completed-fit PhiRL mode was kept explicitly retrospective because its partition and Gaussian parameters use the final three quarters. The cutoff-causal mode refit the exact PhiRL pipeline using only the first quarter and passed separate suffix-deletion, shuffle, replacement, scaling, split, and replay audits.

The target remains exactly `Y=I(H>0.9)`: contemporaneous exact H classifies it with accuracy 1.0 and leaves no unrestricted incremental information for PhiRL. The supplied H baseline uses only first-quarter H history and is therefore a historical predictor, not the unavailable contemporaneous future H. Its apparently near-perfect raw median accuracy (0.982906 in candidate 2 and 0.984510 in candidate 3) is the majority-class effect: balanced accuracy is approximately 0.50 and specificity is 0. The frozen target prevalence and strict pre-onset eligibility results below are therefore central to interpretation.

## Frozen question

Does completed-fit first-quarter PhiRL reconstruct the paper-like accuracy advantage, and does any advantage survive a first-quarter-only source fit and all six directed prospective gates in both candidates?

## Inputs and provenance

- Exactly 100 shared S13Y matrix identities and both frozen candidates 2 and 3; no candidate pooling for primary inference.
- Exactly 200 frozen raw C1 trajectories, the frozen molecular `H>0.9` labels, and the exact S13Y PhiRL completed-fit values.
- Candidate 2: `h=0.6031526490073492`, first daughter, trimmed new entrants. Candidate 3: `h=0.5613315384859516`, random nonempty daughter, trimmed new entrants.
- Original paper SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Pre-outcome repository lock and pushed commit are recorded in `preoutcome_design_lock.json` and `provenance_manifest.json`.

## Detailed methods

### Pre-outcome tensor, split, and architecture lock

The paper specifies the first 25%, remaining 75%, an MLP, 80/20 runs, ten seeds, accuracy, and four baselines, but no tensor orientation, masking, validation, scaling, architecture, optimizer, or flux formula. Before prediction outcomes, S16 froze one reconstruction: for each run of length `T`, `c=floor(0.25T)`, input observations `0..c-1`, and target labels `c..T-1`. Original molecular-step order is right-padded to 367 input and 1,101 target positions with explicit channel/time/target masks; no interpolation, resampling, or truncation occurs.

Every learned family uses the same 288,789-parameter float64 MLP: a shared per-time 200-to-8 encoder over 100 scaled values plus 100 channel-mask values, flattened original order plus the time mask, two 64-unit ReLU/dropout hidden layers, and 1,101 output logits. AdamW, regularization, full-batch masked BCE, the training-only validation rule, 120-epoch ceiling, 15-epoch patience, and all seeds are identical. No hyperparameter or architecture tournament occurred.

Exactly ten outcome-blind, independently seeded matrix splits assign 64 fit, 16 validation, and 20 test matrices. The split identities are identical across candidates, modes, and features. Scaling is fit only on valid input cells of the 64 fit matrices; validation, test, padding, and all post-cutoff observations are excluded.

### Feature and target construction

- `PHIRL_EMERGENCE`: completed-fit first-quarter values in retrospective mode; one exact first-quarter-only PhiRL refit in cutoff mode.
- `COMPOSITION_CHANGE_L2`: scalar Euclidean change of raw relative composition, matching the frozen ordinary-stability control.
- `RAW_COUNTS`: the exact 100 molecular counts.
- `NET_COUNT_FLUX`: the 100-dimensional adjacent count increment, including fission-boundary changes.
- `EXACT_H_HISTORY`: the frozen incoming H sequence, including S13Y's duplicated first adjacent value.
- `MAJORITY_DUMMY`: the fit-subset target prevalence, with no validation/test label access.

The target is the frozen molecular same-state `Y=I(H>0.9)` suffix. Accuracy is the primary micro valid-position metric; matrix-macro accuracy, AUROC, AUPRC, Brier, ten-bin ECE, balanced accuracy, sensitivity, and specificity are secondary. Metrics are also reported by exact target offset. The strict pre-onset risk set includes only runs with no positive input-quarter label and stops at the first future positive inclusive.

### Uncertainty and interpretation gates

Ten-split means, medians, sample SDs, and Student-t intervals are reported. Paper-like two-sided Mann–Whitney tests are retained beside paired Wilcoxon diagnostics. The stronger accuracy comparison resamples unique test matrix identities 4,096 times, retaining all out-of-sample repetitions for each selected matrix. `PROSPECTIVE_PREDICTION_SUPPORTED` requires all six directed gates independently in both candidates; no completed-fit result can enter those gates.

## Commands

```bash
PYTHONPATH=src python scripts/e01/freeze_s16_prediction_design.py
PYTHONPATH=src pytest -q tests/e01/test_s16_prediction_reconstruction.py
PYTHONPATH=src ruff check src/e01_prediction_reconstruction scripts/e01/freeze_s16_prediction_design.py scripts/e01/run_s16_prediction_reconstruction.py tests/e01/test_s16_prediction_reconstruction.py
PYTHONPATH=src python -m compileall -q src/e01_prediction_reconstruction scripts/e01/freeze_s16_prediction_design.py scripts/e01/run_s16_prediction_reconstruction.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s16_prediction_reconstruction.py --output-root /artifacts/research_steps/S16
```

CPU float64 is authoritative; no GPU was used, so no CPU/GPU equivalence claim is needed. No simulator, network call, package installer, author contact, or S17 operation occurred.

## Results

### Candidate-specific split accuracy

| candidateId       | modeId                                             | featureId             |     mean |   median |   sampleStd |   lower95 |   upper95 |
|:------------------|:---------------------------------------------------|:----------------------|---------:|---------:|------------:|----------:|----------:|
| S12F-CANDIDATE-02 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | COMPOSITION_CHANGE_L2 | 0.982147 | 0.982906 |  0.00374597 |  0.979468 |  0.984827 |
| S12F-CANDIDATE-02 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | EXACT_H_HISTORY       | 0.982155 | 0.982906 |  0.00375447 |  0.979469 |  0.984841 |
| S12F-CANDIDATE-02 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | MAJORITY_DUMMY        | 0.982549 | 0.982978 |  0.00343402 |  0.980092 |  0.985005 |
| S12F-CANDIDATE-02 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | NET_COUNT_FLUX        | 0.982189 | 0.982906 |  0.00368152 |  0.979556 |  0.984823 |
| S12F-CANDIDATE-02 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | PHIRL_EMERGENCE       | 0.982147 | 0.982906 |  0.00374597 |  0.979468 |  0.984827 |
| S12F-CANDIDATE-02 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | RAW_COUNTS            | 0.982197 | 0.982797 |  0.00367295 |  0.97957  |  0.984825 |
| S12F-CANDIDATE-03 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | COMPOSITION_CHANGE_L2 | 0.984384 | 0.98451  |  0.00194889 |  0.98299  |  0.985778 |
| S12F-CANDIDATE-03 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | EXACT_H_HISTORY       | 0.984357 | 0.98451  |  0.00196546 |  0.982951 |  0.985763 |
| S12F-CANDIDATE-03 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | MAJORITY_DUMMY        | 0.984608 | 0.984839 |  0.00193479 |  0.983224 |  0.985992 |
| S12F-CANDIDATE-03 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | NET_COUNT_FLUX        | 0.984158 | 0.98451  |  0.00226774 |  0.982536 |  0.98578  |
| S12F-CANDIDATE-03 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | PHIRL_EMERGENCE       | 0.984349 | 0.98451  |  0.00197069 |  0.982939 |  0.985759 |
| S12F-CANDIDATE-03 | CUTOFF_CAUSAL_FIRST_QUARTER_ONLY                   | RAW_COUNTS            | 0.984259 | 0.98451  |  0.00210349 |  0.982754 |  0.985764 |
| S12F-CANDIDATE-02 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | COMPOSITION_CHANGE_L2 | 0.982147 | 0.982906 |  0.00374597 |  0.979468 |  0.984827 |
| S12F-CANDIDATE-02 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | EXACT_H_HISTORY       | 0.982155 | 0.982906 |  0.00375447 |  0.979469 |  0.984841 |
| S12F-CANDIDATE-02 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | MAJORITY_DUMMY        | 0.982549 | 0.982978 |  0.00343402 |  0.980092 |  0.985005 |
| S12F-CANDIDATE-02 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | NET_COUNT_FLUX        | 0.982189 | 0.982906 |  0.00368152 |  0.979556 |  0.984823 |
| S12F-CANDIDATE-02 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | PHIRL_EMERGENCE       | 0.982175 | 0.982906 |  0.0037026  |  0.979527 |  0.984824 |
| S12F-CANDIDATE-02 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | RAW_COUNTS            | 0.982197 | 0.982797 |  0.00367295 |  0.97957  |  0.984825 |
| S12F-CANDIDATE-03 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | COMPOSITION_CHANGE_L2 | 0.984384 | 0.98451  |  0.00194889 |  0.98299  |  0.985778 |
| S12F-CANDIDATE-03 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | EXACT_H_HISTORY       | 0.984357 | 0.98451  |  0.00196546 |  0.982951 |  0.985763 |
| S12F-CANDIDATE-03 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | MAJORITY_DUMMY        | 0.984608 | 0.984839 |  0.00193479 |  0.983224 |  0.985992 |
| S12F-CANDIDATE-03 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | NET_COUNT_FLUX        | 0.984158 | 0.98451  |  0.00226774 |  0.982536 |  0.98578  |
| S12F-CANDIDATE-03 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | PHIRL_EMERGENCE       | 0.984344 | 0.98451  |  0.00198931 |  0.982921 |  0.985767 |
| S12F-CANDIDATE-03 | RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE | RAW_COUNTS            | 0.984259 | 0.98451  |  0.00210349 |  0.982754 |  0.985764 |

### Secondary metrics across all feature families

The table gives the median across the ten frozen splits. AUPRC tracks the 98% positive prevalence, while AUROC and balanced accuracy remain approximately chance. Sensitivity is essentially 1 and specificity is 0 for every family, showing that the high raw accuracy is not discriminative prediction. Exact split-level values, means, SDs, and 95% split-t intervals are in `split_metrics.csv` and `split_metric_summary.csv`.

| Candidate | Mode | Feature | AUROC | AUPRC | Brier | ECE | Balanced accuracy | Sensitivity | Specificity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 02 | cutoff-causal | composition change | 0.468602 | 0.980779 | 0.017187 | 0.005408 | 0.499943 | 0.999887 | 0.000000 |
| 02 | cutoff-causal | H history | 0.475771 | 0.980862 | 0.017225 | 0.003341 | 0.499943 | 0.999887 | 0.000000 |
| 02 | cutoff-causal | dummy | 0.500000 | 0.982978 | 0.016734 | 0.004302 | 0.500000 | 1.000000 | 0.000000 |
| 02 | cutoff-causal | flux | 0.473945 | 0.981455 | 0.017252 | 0.003993 | 0.499943 | 0.999887 | 0.000000 |
| 02 | cutoff-causal | PhiRL | 0.477849 | 0.981531 | 0.017212 | 0.004383 | 0.499943 | 0.999887 | 0.000000 |
| 02 | cutoff-causal | raw counts | 0.494814 | 0.981714 | 0.017657 | 0.008529 | 0.499962 | 0.999882 | 0.000000 |
| 02 | completed-fit | composition change | 0.468602 | 0.980779 | 0.017187 | 0.005408 | 0.499943 | 0.999887 | 0.000000 |
| 02 | completed-fit | H history | 0.475771 | 0.980862 | 0.017225 | 0.003341 | 0.499943 | 0.999887 | 0.000000 |
| 02 | completed-fit | dummy | 0.500000 | 0.982978 | 0.016734 | 0.004302 | 0.500000 | 1.000000 | 0.000000 |
| 02 | completed-fit | flux | 0.473945 | 0.981455 | 0.017252 | 0.003993 | 0.499943 | 0.999887 | 0.000000 |
| 02 | completed-fit | PhiRL | 0.475036 | 0.981258 | 0.017215 | 0.005670 | 0.499943 | 0.999887 | 0.000000 |
| 02 | completed-fit | raw counts | 0.494814 | 0.981714 | 0.017657 | 0.008529 | 0.499962 | 0.999882 | 0.000000 |
| 03 | cutoff-causal | composition change | 0.507594 | 0.984861 | 0.015529 | 0.003180 | 0.500000 | 1.000000 | 0.000000 |
| 03 | cutoff-causal | H history | 0.509199 | 0.985113 | 0.015569 | 0.002600 | 0.500000 | 1.000000 | 0.000000 |
| 03 | cutoff-causal | dummy | 0.500000 | 0.984839 | 0.014932 | 0.001368 | 0.500000 | 1.000000 | 0.000000 |
| 03 | cutoff-causal | flux | 0.510019 | 0.985508 | 0.015509 | 0.003693 | 0.500000 | 1.000000 | 0.000000 |
| 03 | cutoff-causal | PhiRL | 0.510136 | 0.985025 | 0.015559 | 0.002611 | 0.500000 | 1.000000 | 0.000000 |
| 03 | cutoff-causal | raw counts | 0.500931 | 0.984629 | 0.016068 | 0.006337 | 0.500000 | 1.000000 | 0.000000 |
| 03 | completed-fit | composition change | 0.507594 | 0.984861 | 0.015529 | 0.003180 | 0.500000 | 1.000000 | 0.000000 |
| 03 | completed-fit | H history | 0.509199 | 0.985113 | 0.015569 | 0.002600 | 0.500000 | 1.000000 | 0.000000 |
| 03 | completed-fit | dummy | 0.500000 | 0.984839 | 0.014932 | 0.001368 | 0.500000 | 1.000000 | 0.000000 |
| 03 | completed-fit | flux | 0.510019 | 0.985508 | 0.015509 | 0.003693 | 0.500000 | 1.000000 | 0.000000 |
| 03 | completed-fit | PhiRL | 0.507638 | 0.985122 | 0.015576 | 0.002521 | 0.500000 | 1.000000 | 0.000000 |
| 03 | completed-fit | raw counts | 0.500931 | 0.984629 | 0.016068 | 0.006337 | 0.500000 | 1.000000 | 0.000000 |

### Paired cutoff-causal PhiRL accuracy contrasts

| candidateId       | comparatorFeatureId   |   referenceMedianAccuracy |   comparatorMedianAccuracy |   medianPairedAccuracyDifference |   pairedDifferenceLower95AcrossSplits |   positiveSplitDifferenceCount |   paperLikeMannWhitneyTwoSidedP |
|:------------------|:----------------------|--------------------------:|---------------------------:|---------------------------------:|--------------------------------------:|-------------------------------:|--------------------------------:|
| S12F-CANDIDATE-02 | MAJORITY_DUMMY        |                  0.982906 |                   0.982978 |                     -0.000111434 |                          -0.00103048  |                              0 |                        0.596287 |
| S12F-CANDIDATE-02 | COMPOSITION_CHANGE_L2 |                  0.982906 |                   0.982906 |                      0           |                           0           |                              0 |                        1        |
| S12F-CANDIDATE-02 | RAW_COUNTS            |                  0.982906 |                   0.982797 |                      0           |                          -0.000176414 |                              1 |                        0.820263 |
| S12F-CANDIDATE-02 | NET_COUNT_FLUX        |                  0.982906 |                   0.982906 |                      0           |                          -0.000137354 |                              0 |                        0.879424 |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY       |                  0.982906 |                   0.982906 |                      0           |                          -2.50974e-05 |                              0 |                        1        |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY        |                  0.98451  |                   0.984839 |                      0           |                          -0.000796285 |                              0 |                        0.732956 |
| S12F-CANDIDATE-03 | COMPOSITION_CHANGE_L2 |                  0.98451  |                   0.98451  |                      0           |                          -8.84347e-05 |                              0 |                        0.90945  |
| S12F-CANDIDATE-03 | RAW_COUNTS            |                  0.98451  |                   0.98451  |                      0           |                          -5.42648e-05 |                              2 |                        0.969759 |
| S12F-CANDIDATE-03 | NET_COUNT_FLUX        |                  0.98451  |                   0.98451  |                      0           |                          -0.000121913 |                              4 |                        0.909518 |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY       |                  0.98451  |                   0.98451  |                      0           |                          -2.51535e-05 |                              0 |                        1        |

### Test target prevalence

| candidateId       |   meanTestPrevalence |   minTestPrevalence |   maxTestPrevalence |   meanAlreadyPositiveInputMatrices |
|:------------------|---------------------:|--------------------:|--------------------:|-----------------------------------:|
| S12F-CANDIDATE-02 |             0.982549 |            0.977235 |            0.98635  |                                 20 |
| S12F-CANDIDATE-03 |             0.984608 |            0.981209 |            0.987901 |                                 20 |

### Strict pre-onset audit

| candidateId       | featureId       |   repetitionId |   eligibleRunCount |   excludedAlreadyPositiveInputRunCount |   validTargetCount | accuracy   | auroc   | auprc   |
|:------------------|:----------------|---------------:|-------------------:|---------------------------------------:|-------------------:|:-----------|:--------|:--------|
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              0 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              0 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              0 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              1 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              1 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              1 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              2 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              2 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              2 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              3 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              3 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              3 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              4 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              4 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              4 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              5 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              5 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              5 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              6 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              6 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              6 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              7 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              7 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              7 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              8 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              8 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              8 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | PHIRL_EMERGENCE |              9 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | EXACT_H_HISTORY |              9 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-02 | MAJORITY_DUMMY  |              9 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              0 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              0 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              0 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              1 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              1 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              1 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              2 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              2 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              2 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              3 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              3 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              3 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              4 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              4 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              4 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              5 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              5 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              5 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              6 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              6 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              6 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              7 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              7 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              7 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              8 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              8 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              8 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | PHIRL_EMERGENCE |              9 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | EXACT_H_HISTORY |              9 |                  0 |                                     20 |                  0 |            |         |         |
| S12F-CANDIDATE-03 | MAJORITY_DUMMY  |              9 |                  0 |                                     20 |                  0 |            |         |         |

No test matrix in either candidate was pre-onset at the cutoff: every one of the 20 test matrices in every split already had at least one positive input-quarter label. Pre-onset-only accuracy, AUROC, and AUPRC are therefore correctly unavailable rather than silently estimated on a post-onset population.

### Per-time-position performance

`per_time_position_metrics.parquet` contains all 235,644 exact candidate/mode/feature/split/offset rows. For cutoff-causal PhiRL, the median accuracy across repetitions at nearly every sufficiently populated offset was 1.0 because the model predicted the positive majority. Candidate 2 had a median 20 valid matrices per split through offsets 0–249, 17.3 through offsets 250–749, and only 1.5 in the sparse 900+ tail; candidate 3 had 20, 18.0, and 1.2, respectively. Tail minima reaching 0 are consequently based on very few long trajectories and do not rescue the zero-specificity result.

### Completed-fit versus cutoff-only PhiRL fitting

| candidateId       |   trajectoryCount |   medianSharedValues |   medianCompletedCutoffSpearman |   medianAbsoluteDifference |   medianPartitionARI |   exactReplayCount |
|:------------------|------------------:|---------------------:|--------------------------------:|---------------------------:|---------------------:|-------------------:|
| S12F-CANDIDATE-02 |               100 |                221   |                        0.378872 |                   0.497013 |            0.0212114 |                100 |
| S12F-CANDIDATE-03 |               100 |                238.5 |                        0.398478 |                   0.529814 |            0.0135474 |                100 |

## Decision

- Retrospective completed-fit classification: `NOT_SUPPORTED_WITHIN_TESTED_SCOPE`.
- Prospective classification: `NOT_SUPPORTED_WITHIN_TESTED_SCOPE`.
- Candidate retrospective gates: `{"S12F-CANDIDATE-02": false, "S12F-CANDIDATE-03": false}`.
- Candidate prospective gates: `{"S12F-CANDIDATE-02": {"gate1BeatsDummy": false, "gate2BeatsCompositionRawFlux": false, "gate3AddsBeyondHAndStability": false, "gate5DirectionalMatrixUncertainty": false, "gate6LeakageReplayCalibration": false}, "S12F-CANDIDATE-03": {"gate1BeatsDummy": false, "gate2BeatsCompositionRawFlux": false, "gate3AddsBeyondHAndStability": false, "gate5DirectionalMatrixUncertainty": false, "gate6LeakageReplayCalibration": false}}`.
- Leakage, split isolation, source/model replay, and ECE checks passed. Calibration gate 6 still failed in each candidate because cutoff PhiRL Brier loss (medians 0.017212 and 0.015559) was worse than the dummy (0.016734 and 0.014932).
- Contemporaneous exact-H target accuracy is 1.0 by construction; unrestricted increment beyond contemporaneous exact H is zero.
- Neither prediction mode supplies causal-control evidence.

## Validation

PASS: 32/32 checks. The validation artifact records every named check, including 200 matrix/candidate payloads, 600 suffix variants, matrix-only split isolation, training-only scaling, exact label identity, same architecture and seed rules, source/model replay, required-artifact presence, immutable-prior postchecks, zero trajectories, and the S17 stop boundary. A separate post-run audit matched all 33 recorded artifact hashes and all 130,315 unique out-of-sample targets to frozen S13Y labels with zero missing or mismatched rows.

## Compute

S16 measured `0.476168` scientific CPU-hours and `0.477132` total process CPU-hours. The new-scientific-compute ledger retains `100.523832` hours under the 105-hour combined ceiling, before the separately protected four-hour validation/artifact reserve.

## Figures and machine-readable artifacts

Four inspected figures reconstruct Figure 5, compare retrospective and cutoff modes, expose calibration/prevalence, and show per-time/pre-onset behavior. Parquet and CSV artifacts retain every split metric, test prediction, training/scaling record, cutoff source fit, suffix audit, per-time result, paired comparison, and matrix-cluster bootstrap.

## Caveats, blockers, failed assumptions, and limitations

- The exact contemporaneous target-defining H is unavailable in the future suffix at prediction time; the mandatory H model receives only first-quarter H history. These are distinct facts and are reported separately.
- Completed-fit PhiRL partitions and Gaussian parameters use the final 75%; that mode is retrospective prediction resemblance only.
- The paper's tensor layout, validation, scaling, architecture, flux definition, weighting, and seed hierarchy are unavailable. S16 is one frozen coherent reconstruction, not author-code identity.
- The molecular adjacent-H target may have a prevalence unlike the paper's Figure 5 baseline; accuracy must be interpreted beside prevalence, balanced accuracy, AUPRC, calibration, and pre-onset eligibility.
- Repeated test splits overlap. Split-t and paper-like Mann–Whitney diagnostics are therefore accompanied by paired matrix-cluster bootstrap intervals.
- SciPy emitted one expected numerical warning while standardizing an all-tie Wilcoxon contrast; the retained two-sided diagnostic is 1.0, and the primary accuracy, paired differences, and matrix-cluster bootstrap outputs are unaffected.
- Candidate pooling is secondary descriptive only and cannot rescue either candidate.
- No alternative split proportions were searched, so E01-C029 is not evaluated.
- Prediction alone cannot establish causal control, and S17 was not started.

## Provenance

`preoutcome_design_lock.json` records the pushed design commit. `input_manifest.json` records before/after hashes for all prior step artifacts and 200 S13Y raw trajectories. `provenance_manifest.json` records the exact runtime, code hashes, command, numeric policy, and compute ledger. Repository source remains in Git.

## Recommended next action

Return control to the Chief Scientist workflow. Keep S17 queued and inactive until separately instructed. Carry forward the separate retrospective/prospective verdicts, exact-H boundary, target prevalence, pre-onset audit, and all S01–S15 classifications. Do not begin S17 in this execution.
