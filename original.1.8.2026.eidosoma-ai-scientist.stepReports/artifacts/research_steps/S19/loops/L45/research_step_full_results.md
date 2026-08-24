# S19-L45 — PhiID Incremental Value for a Hereditary Episode

## Chief/human handoff

- **Step:** `E01-S19-L45-PHI-INCREMENTAL-VALUE-FOR-HEREDITARY-EPISODE-v1.0.0`
- **Status:** complete under the extended L19–L65 autonomous sequence.
- **Classifications:** `PAST_ONLY_PHI_NOT_INCREMENTAL_FOR_HEREDITARY_EPISODE`, `PHI_PROCESS_NON_SUPPORT`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Validation:** immutable L44-and-earlier baseline; L17/L18 source-equivalence fixtures; exact 280-state/trajectory replay; two prospectively separated temporal modes; exact prefix suffix-invariance sentinels; source-defined component identities; train-only preprocessing; 4,096 catalytic-matrix bootstraps; 512 feature and 512 target permutations; exact feature/model/table regeneration; storage and artifact hashes.
- **Recommended next action:** `L46_FUNCTIONAL_HEREDITARY_REGIME_TRANSITION_AUDIT`.

## Frozen question

Does source-defined information dynamics add held-out probability information about the frozen `NEW_HEREDITARY_EPISODE_RUN3` empirical committor beyond direct inheritance frequency, current streak, fission opportunities, adjacent parent–daughter and molecular H, composition change, mass, phase and elapsed selected-clock time?

The target, candidates, 280 matrix/state identities, F12 horizon, three-inheritance certification, metric identities, two feature summaries, ridge-binomial model, training cohort, evaluation cohorts, metrics, bootstraps and null controls were frozen before Phi outcomes. `PAST_ONLY_PREFIX_FIT` uses observations available at the landmark only. `RETROSPECTIVE_COMPLETED_TRAJECTORY_FIT` is explicitly future-dependent and cannot support prospective language.

## Anchor results

### Held-out model metrics

| evaluationCohort   | candidateId       | modelId                   |   states |    qBrier |   matrixBinomialLogLoss |     spearman |   calibrationIntercept |   calibrationSlope |
|:-------------------|:------------------|:--------------------------|---------:|----------:|------------------------:|-------------:|-----------------------:|-------------------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 | DIRECT_CONTROLS           |       41 | 0.0193986 |                0.562623 |  -0.283934   |              1.35777   |         -0.701603  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | DIRECT_PLUS_COMPLETED_PHI |       41 | 0.0329532 |                0.592818 |  -0.166231   |              0.858593  |         -0.0983482 |
| L28_VALIDATION     | S12F-CANDIDATE-02 | DIRECT_PLUS_PAST_PHI      |       41 | 0.0194221 |                0.562477 |  -0.264245   |              1.21991   |         -0.540584  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | INHERITANCE_BASELINE      |       41 | 0.0112417 |                0.527695 |  -0.00897369 |              0.0993586 |          0.852886  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PAST_PHI_ONLY             |       41 | 0.012647  |                0.532183 |  -0.374978   |              2.12331   |         -1.71033   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | TRAINING_PRIOR            |       41 | 0.011138  |                0.52725  | nan          |            nan         |        nan         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | DIRECT_CONTROLS           |       47 | 0.0185666 |                0.515987 |   0.52093    |              0.533517  |          0.340779  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | DIRECT_PLUS_COMPLETED_PHI |       47 | 0.0341611 |                0.98478  |   0.298451   |              0.732896  |          0.0815508 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | DIRECT_PLUS_PAST_PHI      |       47 | 0.0199553 |                0.519525 |   0.503816   |              0.553186  |          0.314109  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | INHERITANCE_BASELINE      |       47 | 0.0151758 |                0.512581 |   0.205134   |              0.636145  |          0.20421   |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PAST_PHI_ONLY             |       47 | 0.0229765 |                0.533938 |  -0.155643   |              0.771366  |          0.0316473 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | TRAINING_PRIOR            |       47 | 0.0126116 |                0.506828 | nan          |            nan         |        nan         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | DIRECT_CONTROLS           |       36 | 0.0149956 |                0.492791 |  -0.405045   |              1.17074   |         -0.43778   |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | DIRECT_PLUS_COMPLETED_PHI |       36 | 0.0120119 |                0.483304 |   0.0128708  |              0.62888   |          0.231745  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | DIRECT_PLUS_PAST_PHI      |       36 | 0.0172863 |                0.495225 |  -0.282772   |              0.894919  |         -0.0973496 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | INHERITANCE_BASELINE      |       36 | 0.0105286 |                0.478575 |   0.0269001  |              1.06967   |         -0.316145  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PAST_PHI_ONLY             |       36 | 0.0110317 |                0.480078 |  -0.150718   |              0.807137  |          0.0122998 |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | TRAINING_PRIOR            |       36 | 0.010503  |                0.478718 | nan          |            nan         |        nan         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | DIRECT_CONTROLS           |       37 | 0.0189126 |                0.532952 |   0.439308   |             -0.0652623 |          1.041     |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | DIRECT_PLUS_COMPLETED_PHI |       37 | 0.0221087 |                0.544029 |   0.352774   |              0.275038  |          0.618772  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | DIRECT_PLUS_PAST_PHI      |       37 | 0.0184555 |                0.5321   |   0.443101   |             -0.0347543 |          1.00447   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | INHERITANCE_BASELINE      |       37 | 0.0165663 |                0.526889 |   0.496681   |             -0.5294    |          1.64626   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PAST_PHI_ONLY             |       37 | 0.0258715 |                0.551479 |  -0.113798   |              0.843709  |         -0.10189   |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | TRAINING_PRIOR            |       37 | 0.0237332 |                0.546621 | nan          |            nan         |        nan         |

### Incremental comparisons

| evaluationCohort   | candidateId       | comparisonId                      | baselineModelId   | augmentedModelId          |   qBrierGain |   qBrierGainLower95 |   qBrierGainUpper95 |   logLossGain |   logLossGainLower95 |   logLossGainUpper95 |   spearmanGain |   spearmanGainLower95 |   spearmanGainUpper95 |
|:-------------------|:------------------|:----------------------------------|:------------------|:--------------------------|-------------:|--------------------:|--------------------:|--------------:|---------------------:|---------------------:|---------------:|----------------------:|----------------------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 | PAST_INCREMENTAL_OVER_DIRECT      | DIRECT_CONTROLS   | DIRECT_PLUS_PAST_PHI      | -2.35415e-05 |        -0.00193982  |         0.00178253  |   0.00014612  |          -0.00760678 |          0.00851901  |     0.0196898  |            -0.183777  |             0.21061   |
| L28_VALIDATION     | S12F-CANDIDATE-02 | COMPLETED_INCREMENTAL_OVER_DIRECT | DIRECT_CONTROLS   | DIRECT_PLUS_COMPLETED_PHI | -0.0135546   |        -0.0354444   |         0.00230206  |  -0.0301949   |          -0.0808079  |          0.00757899  |     0.117703   |            -0.188915  |             0.445273  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PAST_PHI_OVER_PRIOR               | TRAINING_PRIOR    | PAST_PHI_ONLY             | -0.00150899  |        -0.00334594  |        -0.00034261  |  -0.00493307  |          -0.0114171  |         -0.000973803 |   nan          |           nan         |           nan         |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PAST_INCREMENTAL_OVER_DIRECT      | DIRECT_CONTROLS   | DIRECT_PLUS_PAST_PHI      | -0.00138871  |        -0.00337437  |         0.000191869 |  -0.00353784  |          -0.00864663 |          0.000650004 |    -0.0171138  |            -0.102946  |             0.0570605 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | COMPLETED_INCREMENTAL_OVER_DIRECT | DIRECT_CONTROLS   | DIRECT_PLUS_COMPLETED_PHI | -0.0155944   |        -0.0510105   |         0.00703555  |  -0.468793    |          -1.40454    |          0.0107746   |    -0.222479   |            -0.518326  |             0.0533393 |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PAST_PHI_OVER_PRIOR               | TRAINING_PRIOR    | PAST_PHI_ONLY             | -0.0103649   |        -0.0261837   |        -0.00162419  |  -0.0271103   |          -0.0679096  |         -0.00439051  |   nan          |           nan         |           nan         |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PAST_INCREMENTAL_OVER_DIRECT      | DIRECT_CONTROLS   | DIRECT_PLUS_PAST_PHI      | -0.00229072  |        -0.00958172  |         0.00238995  |  -0.00243367  |          -0.0180875  |          0.00951566  |     0.122273   |            -0.109845  |             0.386264  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | COMPLETED_INCREMENTAL_OVER_DIRECT | DIRECT_CONTROLS   | DIRECT_PLUS_COMPLETED_PHI |  0.00298377  |        -0.00220988  |         0.00939189  |   0.00948793  |          -0.00623631 |          0.0288697   |     0.417916   |             0.0197282 |             0.828643  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PAST_PHI_OVER_PRIOR               | TRAINING_PRIOR    | PAST_PHI_ONLY             | -0.00052875  |        -0.00216883  |         0.00125114  |  -0.00136087  |          -0.00684826 |          0.0048825   |   nan          |           nan         |           nan         |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PAST_INCREMENTAL_OVER_DIRECT      | DIRECT_CONTROLS   | DIRECT_PLUS_PAST_PHI      |  0.000457101 |        -0.000336727 |         0.00126218  |   0.000852123 |          -0.00232987 |          0.00391059  |     0.00379327 |            -0.0863232 |             0.0791209 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | COMPLETED_INCREMENTAL_OVER_DIRECT | DIRECT_CONTROLS   | DIRECT_PLUS_COMPLETED_PHI | -0.00319608  |        -0.00805749  |         0.00076249  |  -0.0110765   |          -0.0266698  |          0.00172359  |    -0.0865339  |            -0.311908  |             0.106252  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PAST_PHI_OVER_PRIOR               | TRAINING_PRIOR    | PAST_PHI_ONLY             | -0.00213838  |        -0.00637569  |         0.00147453  |  -0.00485825  |          -0.0154197  |          0.00477855  |   nan          |           nan         |           nan         |

### Permutation controls

| evaluationCohort   | candidateId       | controlId                      |   observedQBrierGain |   observedLogLossGain |   qBrierPermutationP |   logLossPermutationP |   qBrierPermutationPHolm |   logLossPermutationPHolm |
|:-------------------|:------------------|:-------------------------------|---------------------:|----------------------:|---------------------:|----------------------:|-------------------------:|--------------------------:|
| L28_VALIDATION     | S12F-CANDIDATE-02 | DEVELOPMENT_TARGET_PERMUTATION |         -2.35415e-05 |           0.00014612  |           0.214425   |            0.300195   |                0.479532  |                 0.452242  |
| L28_VALIDATION     | S12F-CANDIDATE-02 | PAST_PHI_FEATURE_PERMUTATION   |         -2.35415e-05 |           0.00014612  |           0.0838207  |            0.0818713  |                0.335283  |                 0.245614  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | DEVELOPMENT_TARGET_PERMUTATION |         -0.00138871  |          -0.00353784  |           0.495127   |            0.226121   |                0.495127  |                 0.452242  |
| L28_VALIDATION     | S12F-CANDIDATE-03 | PAST_PHI_FEATURE_PERMUTATION   |         -0.00138871  |          -0.00353784  |           0.380117   |            0.321637   |                0.54386   |                 0.358674  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | DEVELOPMENT_TARGET_PERMUTATION |         -0.00229072  |          -0.00243367  |           0.159844   |            0.0896686  |                0.479532  |                 0.269006  |
| L31_CONFIRMATION   | S12F-CANDIDATE-02 | PAST_PHI_FEATURE_PERMUTATION   |         -0.00229072  |          -0.00243367  |           0.194932   |            0.0584795  |                0.54386   |                 0.233918  |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | DEVELOPMENT_TARGET_PERMUTATION |          0.000457101 |           0.000852123 |           0.00584795 |            0.00584795 |                0.0233918 |                 0.0233918 |
| L31_CONFIRMATION   | S12F-CANDIDATE-03 | PAST_PHI_FEATURE_PERMUTATION   |          0.000457101 |           0.000852123 |           0.181287   |            0.179337   |                0.54386   |                 0.358674  |

### Suffix invariance/dependence

| temporalMode                           |   sentinels |   maximumFeatureDifference | exact   |
|:---------------------------------------|------------:|---------------------------:|:--------|
| PAST_ONLY_PREFIX_FIT                   |          24 |                    0       | True    |
| RETROSPECTIVE_COMPLETED_TRAJECTORY_FIT |          24 |                    1.29637 | False   |

### Scientific gates

| gateId                                            | passed   | prospectiveEligible   |
|:--------------------------------------------------|:---------|:----------------------|
| SOURCE_FEATURE_AVAILABILITY                       | True     | True                  |
| PAST_ONLY_SUFFIX_INVARIANCE                       | True     | True                  |
| PAST_PHI_INCREMENTAL_BRIER_AND_LOGLOSS_ALL_GROUPS | False    | True                  |
| PAST_PHI_PERMUTATION_CONTROLS_ALL_GROUPS          | False    | True                  |
| COMPLETED_FIT_FUTURE_DEPENDENCE_DEMONSTRATED      | True     | False                 |
| COMPLETED_FIT_RETROSPECTIVE_INCREMENTAL_ALIGNMENT | False    | False                 |

## Interpretation boundary

PhiID emergence, integrated Phi-r, synergy and downward causation are retained as separate public-source identities. They are computational information-dynamic summaries, not intervention evidence or proof of physical downward causation. Incremental value is required over direct heredity and phase controls; a raw Phi correlation is not sufficient. Completed-fit alignment, if present, is a retrospective description because suffix information changes the fitted prefix values.

The L44 target is an operational three-fission heredity episode, not exact return to a privileged composition and not an author-code replicator label. This loop changes no S18, paper-replication, intervention or causal-control classification.

## Provenance and validation

- Repository lock: `37d76ef7e2d58235be7b929b930d296762c2e59f`.
- Workers: `8`; one numerical-library thread per worker; GPU hours `0`.
- New matrices/trajectories/branch streams: `0/0/0`.
- Frozen states and trajectories: `280`.
- Exact Phi pipeline evaluations across both full passes: `1216`.
- Wall time: `2764.40` seconds.
- S01–S18, V1/V2 and S19-L01–L44 remain unchanged.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/e01/test_s19_l45.py
python -m ruff check src/e01_onset_discovery/heredity_phi_incremental.py scripts/e01/l45_phi_process_worker.py scripts/e01/run_s19_l45_phi_incremental_hereditary_episode.py tests/e01/test_s19_l45.py
python scripts/e01/run_s19_l45_phi_incremental_hereditary_episode.py --prepare-lock
python scripts/e01/run_s19_l45_phi_incremental_hereditary_episode.py
```
