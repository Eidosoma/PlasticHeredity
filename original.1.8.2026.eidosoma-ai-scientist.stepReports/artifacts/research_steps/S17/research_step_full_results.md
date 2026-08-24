# E01/S17 — Max/control/min intervention reconstruction

## Concise top summary

- **Research step ID:** `E01-S17-INTERVENTION-RECONSTRUCTION-v1.0.0` (actual step `S17`).
- **Completion status:** `COMPLETE_DIRECTED_OPTION_2_WITH_RECORDED_CPU_ALLOWANCE_WAIVER`; the fixed scope completed and control returned before S18.
- **Artifacts written:** The complete retained S17 artifact set, including 72 trajectory identities, every candidate score, action/replay/pairing/null/outcome table, Figure 6 and Table 1 reconstructions, validation/provenance/status manifests, and this report.
- **Validation result:** `PASS_WITH_EXPLICIT_HUMAN_WAIVER_OF_RECORDED_CPU_ALLOWANCE_OVERRUN` (33/34 checks passed directly; 1 compute check explicitly waived).
- **Outcome classification:** literal ordering `NOT_SUPPORTED_WITHIN_TESTED_SCOPE`; prospective causal control `NOT_SUPPORTED_WITHIN_TESTED_SCOPE`.
- **Caveats or blockers:** The authors' exact implementation remains unavailable; Y is exactly I(H>0.9); the fixed 72-trajectory scope permits matched-random action-score diagnostics but no random-action outcome arm; common streams cease to be counterfactually identical after paths diverge. The pre-outcome benchmark gate passed, but actual fixed-scope CPU use exceeded the post-S16 allowance by 1.530459 hours; the human explicitly waived the CPU allowance after execution, so the overrun is recorded but nonblocking.
- **Recommended next action:** Return for human review. Keep S18 queued but inactive; do not start S18 or E02 automatically.

## Lay summary

This step carried out the paper's intervention idea literally on 12 new shared catalytic matrices under both frozen simulator candidates. After every fission, each treated run tried adding every molecular type and deleting every present type. Every hypothetical edit was appended to the information available at that moment and the pinned PhiRL pipeline was refit from scratch; no future state was used. The raw highest-scoring action drove the max condition and the raw lowest drove the min condition, while control did nothing. Literal directional resemblance and the much stronger causal-control claim were judged separately. The exact binary outcome remains a stability-threshold label, `Y=I(H>0.9)`. The benchmark authorized execution at a conservative projection of 93.908088 CPU-hours, but heterogeneous scientific units ultimately used at least 102.054290, exceeding the available allocation by 1.530459; this remains visible under the explicit post-execution human waiver.

## Frozen question

Can the paper's literal online max/control/min procedure be executed reproducibly on the fixed 12-matrix scope, and does any max ≥ control ≥ min ordering survive paired prospective and numerical scrutiny in both candidate 2 and candidate 3?

## Inputs and provenance

- Candidate 2: `h=0.6031526490073492`, first daughter, trim newly joined excess, C1 selected-daughter boundary.
- Candidate 3: `h=0.5613315384859516`, random nonempty daughter, otherwise the same trimming and boundary semantics.
- Exactly 12 new catalytic matrices and matched initial states were shared across candidates and conditions; each named stochastic stream was shared until state paths diverged.
- Original paper SHA-256: `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Pushed scoring/tie/seed/pairing/replay/analysis lock and final compute gate are recorded in `preoutcome_design_lock.json`, `runtime_benchmark.json`, `compute_gate.json`, and `provenance_manifest.json`.
- S01–S16 and the pre-existing forensic bundle were hash-baselined and remained unchanged.

## Detailed methods

### Pre-outcome compute gate

One domain-separated candidate-3 max sequence completed all 100 fissions and its control/replays before any scientific matrix was created. The conservative 1.25× projection was **93.908088 CPU-hours** against **100.523832 available CPU-hours** after S16 and the protected four-hour reserve. The gate passed and the measured gate was committed and pushed before scientific outcomes. Actual benchmark-plus-unit CPU use was at least **102.054290 hours**, an overrun of **1.530459 hours**; no scope was reduced after outcomes. The human subsequently waived the allowance, without changing any scientific method or result.

### Literal scorer and action semantics

At each completed fission, the selected unedited daughter was made available to the decision. For each of 100 additions and each deletion of a currently present type, the hypothetical edited daughter was appended after that boundary. The additive-0.5 closure, full CLR with original component 100 dropped, PhiRL active-variable filter, source-confirmed Fiedler partition, regularized Gaussian local PhiID, and source-defined `emergence = synergy + downward causation` were refit on that prefix. The final local emergence value was the score. No no-op entered max/min; an unedited score was retained only as a diagnostic. Exact binary64 ties used a domain-separated SHA-256 rank. Weak gaps never suppressed an action.

The actual C1 trajectory retains one edited post-fission boundary, matching the established selected-boundary convention. A fixed action-schedule replay regenerated every trajectory. Selected and runner-up source fits were rerun at every decision; the complete candidate set was rerun at generations 1, 50, and 100 of every treated trajectory.

### Outcomes and inference

The primary label is the frozen molecular adjacent-incoming `Y=I(H>0.9)`. Persistence is the number of labelled selected-clock observations; probability is their fraction; consistency is Pearson correlation of adjacent binary labels when defined; time to first is the zero-based selected-clock index, with normalized fraction retained only for the paper's percent-form Table 1 comparison. Episode, transition, parent-daughter, action, exposure, gap, partition, condition, replay, and null diagnostics were retained.

All primary summaries keep candidates separate. The 12 shared matrices are the paired unit. Exactly 4,096 domain-separated matrix bootstraps estimate paired-effect intervals. Pooling appears only as secondary description.

## Commands

```bash
PYTHONPATH=src python scripts/e01/freeze_s17_intervention_design.py
PYTHONPATH=src pytest -q tests/e01/test_s17_intervention_reconstruction.py
PYTHONPATH=src ruff check src/e01_intervention_reconstruction scripts/e01/freeze_s17_intervention_design.py scripts/e01/run_s17_intervention_reconstruction.py tests/e01/test_s17_intervention_reconstruction.py
PYTHONPATH=src python -m compileall -q src/e01_intervention_reconstruction scripts/e01/freeze_s17_intervention_design.py scripts/e01/run_s17_intervention_reconstruction.py
PYTHONPATH=src python scripts/e01/run_s17_intervention_reconstruction.py --stage benchmark
# commit and push configs/e01/s17_compute_gate.json
PYTHONPATH=src python scripts/e01/run_s17_intervention_reconstruction.py --stage scientific --workers 8
```

CPU float64 was authoritative. Eight independent candidate/matrix workers and one numerical-library thread per worker were used. No GPU or network access was used; therefore no CPU/GPU equivalence claim is made.

## Dependencies and runtime

The supplied scientific environment was used without installing dependencies: Python 3.13.14, NumPy 2.4.6, pandas 2.3.3, SciPy 1.18.0, and Matplotlib 3.11.1. The pinned repository PhiRL/GARD implementation and frozen safe-lattice source were used directly. Operating-system and repository commit details are recorded in `provenance_manifest.json`.

## Results

### Candidate-specific Table 1 outcomes

| candidateId       | condition   | outcome               |   n |         mean |        median |    sampleStd |       lower95 |      upper95 |
|:------------------|:------------|:----------------------|----:|-------------:|--------------:|-------------:|--------------:|-------------:|
| S12F-CANDIDATE-02 | CONTROL     | persistence           |  12 | 771.25       | 776           | 261.063      | 605.379       | 937.121      |
| S12F-CANDIDATE-02 | CONTROL     | probability           |  12 |   0.980029   |   0.981819    |   0.00952938 |   0.973975    |   0.986084   |
| S12F-CANDIDATE-02 | CONTROL     | consistency           |  12 |   0.12368    |   0.0417142   |   0.238305   |  -0.0277318   |   0.275092   |
| S12F-CANDIDATE-02 | CONTROL     | timeToFirstNormalized |  12 |   0.00223956 |   0.000874891 |   0.00323972 |   0.000181139 |   0.00429798 |
| S12F-CANDIDATE-02 | MAX         | persistence           |  12 | 751.333      | 764           | 240.111      | 598.774       | 903.892      |
| S12F-CANDIDATE-02 | MAX         | probability           |  12 |   0.978607   |   0.979545    |   0.0122874  |   0.9708      |   0.986414   |
| S12F-CANDIDATE-02 | MAX         | consistency           |  12 |   0.175781   |   0.056275    |   0.294722   |  -0.0114768   |   0.363038   |
| S12F-CANDIDATE-02 | MAX         | timeToFirstNormalized |  12 |   0.00218649 |   0.000865052 |   0.00309572 |   0.000219567 |   0.00415342 |
| S12F-CANDIDATE-02 | MIN         | persistence           |  12 | 721.75       | 724           | 238.621      | 570.137       | 873.363      |
| S12F-CANDIDATE-02 | MIN         | probability           |  12 |   0.975188   |   0.980474    |   0.0128678  |   0.967012    |   0.983363   |
| S12F-CANDIDATE-02 | MIN         | consistency           |  12 |   0.135457   |   0.0303107   |   0.283438   |  -0.0446316   |   0.315545   |
| S12F-CANDIDATE-02 | MIN         | timeToFirstNormalized |  12 |   0.00278553 |   0.000856164 |   0.00452073 |  -8.6804e-05  |   0.00565786 |
| S12F-CANDIDATE-03 | CONTROL     | persistence           |  12 | 789.167      | 820           | 270.31       | 617.42        | 960.913      |
| S12F-CANDIDATE-03 | CONTROL     | probability           |  12 |   0.983269   |   0.985692    |   0.0100849  |   0.976861    |   0.989676   |
| S12F-CANDIDATE-03 | CONTROL     | consistency           |  12 |   0.145014   |   0.0899859   |   0.203745   |   0.0155602   |   0.274467   |
| S12F-CANDIDATE-03 | CONTROL     | timeToFirstNormalized |  12 |   0.00146521 |   0           |   0.00247245 |  -0.00010571  |   0.00303613 |
| S12F-CANDIDATE-03 | MAX         | persistence           |  12 | 786          | 798           | 261.103      | 620.103       | 951.897      |
| S12F-CANDIDATE-03 | MAX         | probability           |  12 |   0.980655   |   0.982721    |   0.00974653 |   0.974463    |   0.986848   |
| S12F-CANDIDATE-03 | MAX         | consistency           |  12 |   0.124389   |   0.0783635   |   0.1969     |  -0.000715471 |   0.249493   |
| S12F-CANDIDATE-03 | MAX         | timeToFirstNormalized |  12 |   0.00141295 |   0           |   0.0023335  |  -6.96849e-05 |   0.00289558 |
| S12F-CANDIDATE-03 | MIN         | persistence           |  12 | 769          | 773           | 255.115      | 606.908       | 931.092      |
| S12F-CANDIDATE-03 | MIN         | probability           |  12 |   0.97715    |   0.978812    |   0.0141729  |   0.968145    |   0.986155   |
| S12F-CANDIDATE-03 | MIN         | consistency           |  12 |   0.111975   |   0.0788084   |   0.142671   |   0.0213262   |   0.202623   |
| S12F-CANDIDATE-03 | MIN         | timeToFirstNormalized |  12 |   0.00144896 |   0           |   0.00241445 |  -8.51106e-05 |   0.00298303 |

### Paired max/control/min effects

Positive `MAX_MINUS_CONTROL` and positive `CONTROL_MINUS_MIN` both favor the paper's direction.

| candidateId       | outcome     | comparison        |   pairedMatrixCount |   meanDifference |   medianDifference |   bootstrapLower95 |   bootstrapUpper95 |   wilcoxonTwoSidedP |
|:------------------|:------------|:------------------|--------------------:|-----------------:|-------------------:|-------------------:|-------------------:|--------------------:|
| S12F-CANDIDATE-02 | persistence | MAX_MINUS_CONTROL |                  12 |     -19.9167     |        -4          |       -44.5833     |        2           |           0.339355  |
| S12F-CANDIDATE-02 | persistence | CONTROL_MINUS_MIN |                  12 |      49.5        |        28.5        |        17.7812     |       85.3333      |           0.0170898 |
| S12F-CANDIDATE-02 | probability | MAX_MINUS_CONTROL |                  12 |      -0.00142205 |        -0.00226395 |        -0.00481519 |        0.00213821  |           0.380371  |
| S12F-CANDIDATE-02 | probability | CONTROL_MINUS_MIN |                  12 |       0.00484153 |         0.00408924 |         0.00163336 |        0.00834843  |           0.0161133 |
| S12F-CANDIDATE-03 | persistence | MAX_MINUS_CONTROL |                  12 |      -3.16667    |       -10.5        |       -27.5521     |       21.4167      |           0.969727  |
| S12F-CANDIDATE-03 | persistence | CONTROL_MINUS_MIN |                  12 |      20.1667     |        17          |       -13.0833     |       60.3854      |           0.436035  |
| S12F-CANDIDATE-03 | probability | MAX_MINUS_CONTROL |                  12 |      -0.00261347 |        -0.00147363 |        -0.00581156 |        0.000527234 |           0.266113  |
| S12F-CANDIDATE-03 | probability | CONTROL_MINUS_MIN |                  12 |       0.00611865 |         0.00768046 |         0.0019909  |        0.0101015   |           0.0209961 |

### Action execution and numerical scrutiny

| candidateId       | condition   |   decisionCount |   actionFrequency |   additionCount |   deletionCount |   exactTieFraction |   medianGap |   gapAboveRunnerUpUncertaintyFraction |   matchedRandomCorrectDirectionFraction |   maximumSelectedReplayError |
|:------------------|:------------|----------------:|------------------:|----------------:|----------------:|-------------------:|------------:|--------------------------------------:|----------------------------------------:|-----------------------------:|
| S12F-CANDIDATE-02 | MAX         |            1200 |                 1 |             554 |             646 |          0.0133333 |  0.00317008 |                              0.94     |                                0.998333 |                            0 |
| S12F-CANDIDATE-02 | MIN         |            1200 |                 1 |             958 |             242 |          0.0133333 |  0.0275185  |                              0.925833 |                                1        |                            0 |
| S12F-CANDIDATE-03 | MAX         |            1200 |                 1 |             574 |             626 |          0.0133333 |  0.00209718 |                              0.933333 |                                1        |                            0 |
| S12F-CANDIDATE-03 | MIN         |            1200 |                 1 |             939 |             261 |          0.0141667 |  0.0325891  |                              0.934167 |                                1        |                            0 |

### Generation-level probability trends

| candidateId       | condition   |       slope |   twoSidedP |   firstGenerationMeanProbability |   lastGenerationMeanProbability |
|:------------------|:------------|------------:|------------:|---------------------------------:|--------------------------------:|
| S12F-CANDIDATE-02 | CONTROL     | 0.000229501 |   0.0237152 |                         0.746892 |                        0.994048 |
| S12F-CANDIDATE-02 | MAX         | 0.000116316 |   0.286266  |                         0.734987 |                        1        |
| S12F-CANDIDATE-02 | MIN         | 9.26276e-05 |   0.432893  |                         0.716799 |                        0.992424 |
| S12F-CANDIDATE-03 | CONTROL     | 0.000157505 |   0.0953967 |                         0.755919 |                        0.972222 |
| S12F-CANDIDATE-03 | MAX         | 0.000204362 |   0.0519378 |                         0.755919 |                        0.994444 |
| S12F-CANDIDATE-03 | MIN         | 0.000147354 |   0.128128  |                         0.755919 |                        0.977679 |

### Paper-target comparison

The paper's `±` convention is not identified as SD or SE, so numerical target proximity is descriptive. Full candidate-specific and secondary pooled rows are in `paper_target_comparison.csv`.

| candidateId       | condition   | outcome               |   observedMean |   paperMean |   observedMinusPaper | withinOnePaperReportedDispersion   |
|:------------------|:------------|:----------------------|---------------:|------------:|---------------------:|:-----------------------------------|
| S12F-CANDIDATE-02 | MAX         | persistence           |   751.333      |      874    |         -122.667     | True                               |
| S12F-CANDIDATE-02 | MAX         | probability           |     0.978607   |        0.88 |            0.0986071 | False                              |
| S12F-CANDIDATE-02 | MAX         | consistency           |     0.175781   |        0.52 |           -0.344219  | False                              |
| S12F-CANDIDATE-02 | MAX         | timeToFirstNormalized |     0.00218649 |        0.36 |           -0.357814  | False                              |
| S12F-CANDIDATE-02 | CONTROL     | persistence           |   771.25       |      716    |           55.25      | True                               |
| S12F-CANDIDATE-02 | CONTROL     | probability           |     0.980029   |        0.88 |            0.100029  | False                              |
| S12F-CANDIDATE-02 | CONTROL     | consistency           |     0.12368    |        0.38 |           -0.25632   | False                              |
| S12F-CANDIDATE-02 | CONTROL     | timeToFirstNormalized |     0.00223956 |        0.37 |           -0.36776   | False                              |
| S12F-CANDIDATE-02 | MIN         | persistence           |   721.75       |      559    |          162.75      | False                              |
| S12F-CANDIDATE-02 | MIN         | probability           |     0.975188   |        0.8  |            0.175188  | False                              |
| S12F-CANDIDATE-02 | MIN         | consistency           |     0.135457   |        0.42 |           -0.284543  | False                              |
| S12F-CANDIDATE-02 | MIN         | timeToFirstNormalized |     0.00278553 |        0.4  |           -0.397214  | False                              |
| S12F-CANDIDATE-03 | MAX         | persistence           |   786          |      874    |          -88         | True                               |
| S12F-CANDIDATE-03 | MAX         | probability           |     0.980655   |        0.88 |            0.100655  | False                              |
| S12F-CANDIDATE-03 | MAX         | consistency           |     0.124389   |        0.52 |           -0.395611  | False                              |
| S12F-CANDIDATE-03 | MAX         | timeToFirstNormalized |     0.00141295 |        0.36 |           -0.358587  | False                              |
| S12F-CANDIDATE-03 | CONTROL     | persistence           |   789.167      |      716    |           73.1667    | True                               |
| S12F-CANDIDATE-03 | CONTROL     | probability           |     0.983269   |        0.88 |            0.103269  | False                              |
| S12F-CANDIDATE-03 | CONTROL     | consistency           |     0.145014   |        0.38 |           -0.234986  | False                              |
| S12F-CANDIDATE-03 | CONTROL     | timeToFirstNormalized |     0.00146521 |        0.37 |           -0.368535  | False                              |
| S12F-CANDIDATE-03 | MIN         | persistence           |   769          |      559    |          210         | False                              |
| S12F-CANDIDATE-03 | MIN         | probability           |     0.97715    |        0.8  |            0.17715   | False                              |
| S12F-CANDIDATE-03 | MIN         | consistency           |     0.111975   |        0.42 |           -0.308025  | False                              |
| S12F-CANDIDATE-03 | MIN         | timeToFirstNormalized |     0.00144896 |        0.4  |           -0.398551  | False                              |

## Interpretation gates

- **Literal intervention ordering:** `NOT_SUPPORTED_WITHIN_TESTED_SCOPE`. This asks only whether the raw online max/control/min reconstruction ran exactly and produced the directed aggregate ordering in both simulator candidates.
- **Prospective causal control:** `NOT_SUPPORTED_WITHIN_TESTED_SCOPE`. This separately requires opposite paired effects with intervals excluding zero, adequate action frequency, action separability, numerical stability, cross-candidate agreement, and exclusion of matched-random explanations.
- **Matched-random boundary:** `UNDERDETERMINED_NO_FOURTH_RANDOM_ACTION_ROLLOUT_IN_EXACT_72_SCOPE`. Within-state and displacement-matched score nulls were evaluated, but adding a fourth random-action rollout would violate the exact 72-trajectory lock. Score extremeness cannot by itself exclude random-action outcome effects.
- Exact H determines the contemporaneous binary label. Intervention ordering on this label is not independent evidence of an information increment beyond exact H or ordinary stability.

## Validation

All 34 recorded checks are in `validation.json`; 33 passed directly and the sole nonpassing raw check—the actual CPU allowance after a valid pre-outcome benchmark gate—was explicitly waived by the human. The remaining checks cover the pushed locks, exact scope and pairing, action enumeration, no-op exclusion, source and trajectory replay, suffix-free scoring by construction, label identity, trajectory completion, candidate separation, bootstrap cardinality, prior immutability, artifact completeness, and the S18 stop boundary.

## Caveats, blockers, and limitations

The authors' exact implementation remains unavailable; Y is exactly I(H>0.9); the fixed 72-trajectory scope permits matched-random action-score diagnostics but no random-action outcome arm; common streams cease to be counterfactually identical after paths diverge. The pre-outcome benchmark gate passed, but actual fixed-scope CPU use exceeded the post-S16 allowance by 1.530459 hours; the human explicitly waived the CPU allowance after execution, so the overrun is recorded but nonblocking. The no-action diagnostic never entered selection. Fiedler partitions and Gaussian fits were candidate-action-specific; this is computationally literal but may differ from unavailable author code. The paper does not identify its tie rule, prefix content, seed semantics, whether Table 1 `±` is SD or SE, or its exact intervention-state clock accounting. Exact replay demonstrates software determinism, not causal truth. The 12-matrix design has limited paired uncertainty resolution. Completed-fit S13Y resemblance, S15 association, and S16 prediction non-support remain unchanged and evidentially separate.

## Provenance and artifact map

Every reusable table, figure, manifest, cache identity, repository file hash, runtime quantity, and validation result is listed in `artifact_manifest.json` and `provenance_manifest.json`. Full trajectories remain in `/cache/e01_s17_v1/trajectories`; collectible trajectory identities and hashes are in `trajectory_manifest.parquet`. Every candidate action score is retained in `action_candidate_scores.parquet`. The reconstructed Figure 6 is `figures/figure6_reconstruction.png`; the Table 1 audit is `table1_reconstruction.csv` and is also promoted to the forensic bundle.

## Recommended next action

Return for human review. S18 remains queued but inactive. Do not add another scorer, estimator, simulator, threshold, label, random-action treatment, or method inside S17, and do not start S18 or E02 automatically.
