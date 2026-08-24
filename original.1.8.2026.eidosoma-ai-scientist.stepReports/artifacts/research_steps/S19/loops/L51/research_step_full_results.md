# S19-L51 Full Results — Heredity-Regime Hazard and Renewal Decomposition

## Top summary

- **Research step:** `E01-S19-L51-HEREDITY-REGIME-HAZARD-RENEWAL-DECOMPOSITION-v1.0.0`
- **Completion status:** complete; additive exploratory analysis-only evidence
- **Artifacts written:** exact L50 prefix/branch-sequence replay, six locked stochastic-process models, transition hazards, finite-horizon break/conditional/joint probabilities, 4,096 catalytic-matrix bootstraps, 512 whole-matrix permutations, variance decomposition, seven figures, report and hash manifests
- **Validation:** PASS — immutable S01–L50 evidence; eleven fixtures; exact 800-state and 51,200-branch reconstruction; strict-H identity; source/input/repository/seed locks; two exact analysis passes; report regeneration; runtime, storage and artifact hashes
- **Outcome classification:** `DURATION_DEPENDENT_HEREDITY_REGIME_SWITCHING_IDENTIFIED`, `LONGITUDINAL_STATE_UPDATING_ADDS_INFORMATION`, `REGISTERED_PROCESS_MODELS_DO_NOT_RECONSTRUCT_EMPIRICAL_COMMITTOR`, `RISK_VARIATION_PRIMARILY_BETWEEN_MATRICES`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Lay summary:** L51 asks whether the process behaves like independent inheritance, a two-state Markov switch, a duration-dependent renewal process, a stable matrix-specific propensity, or a propensity that changes along one lineage. It does not create a new replicator label or simulate a new future.
- **Recommended next action:** `L52_PROCESS_ALIGNED_SHOOTING_RESIDUAL_AUDIT` under the bounded autonomous authorization through L65. S20, E02, author contact, Phi variants and interventions remain inactive.

## Frozen question and design

The input is exactly L50: 40 development and 40 validation catalytic matrices, both simulator candidates, five post-fission landmarks, and 64 F12 branches per state. The strict `H>0.9` parent/daughter inheritance sequence and the break-then-three-inherited-fissions event are unchanged. Break probability, resumption conditional on a break, and their joint probability remain separate.

Six complete models were registered before opening L51-derived outcomes. The hierarchy begins with pooled IID and first-order Markov transition probabilities, adds a capped-dwell semi-Markov hazard, then adds matrix-specific transition probabilities learned only from the observed primary-lineage prefix. The *early* matrix model is frozen at fission 20 for all later landmarks; the *current* model updates only from observations available at its landmark. Pooled models are fit only on development matrices. No threshold, duration cap, smoothing strength, model, horizon or candidate was selected by result proximity.

## Heldout one-step transition models

| candidateId       | modelId                          |   matrices |   equalMatrixMeanLogLoss |   equalMatrixMeanBrier |
|:------------------|:---------------------------------|-----------:|-------------------------:|-----------------------:|
| S12F-CANDIDATE-02 | POOLED_SEMIMARKOV_DURATION       |         40 |                0.3170394 |              0.0920276 |
| S12F-CANDIDATE-02 | POOLED_MARKOV                    |         40 |                0.3308374 |              0.0942806 |
| S12F-CANDIDATE-02 | CURRENT_PREFIX_MATRIX_SEMIMARKOV |         40 |                0.3329608 |              0.0967753 |
| S12F-CANDIDATE-02 | EARLY_PREFIX_MATRIX_MARKOV       |         40 |                0.3510885 |              0.0994690 |
| S12F-CANDIDATE-02 | EARLY_PREFIX_MATRIX_SEMIMARKOV   |         40 |                0.3527489 |              0.1010774 |
| S12F-CANDIDATE-02 | POOLED_IID                       |         40 |                0.3554017 |              0.1009253 |
| S12F-CANDIDATE-03 | POOLED_SEMIMARKOV_DURATION       |         40 |                0.3226399 |              0.0943851 |
| S12F-CANDIDATE-03 | CURRENT_PREFIX_MATRIX_SEMIMARKOV |         40 |                0.3379467 |              0.0985714 |
| S12F-CANDIDATE-03 | POOLED_MARKOV                    |         40 |                0.3384623 |              0.0973719 |
| S12F-CANDIDATE-03 | EARLY_PREFIX_MATRIX_MARKOV       |         40 |                0.3633169 |              0.1013361 |
| S12F-CANDIDATE-03 | EARLY_PREFIX_MATRIX_SEMIMARKOV   |         40 |                0.3656163 |              0.1042135 |
| S12F-CANDIDATE-03 | POOLED_IID                       |         40 |                0.3661029 |              0.1050146 |

### Registered transition comparisons

| candidateId       | comparisonId                  | modelId                          | referenceModelId               |   matrices |   logLossImprovement |   logLossImprovementLower95 |   logLossImprovementUpper95 |   fractionBootstrapPositive |
|:------------------|:------------------------------|:---------------------------------|:-------------------------------|-----------:|---------------------:|----------------------------:|----------------------------:|----------------------------:|
| S12F-CANDIDATE-02 | DURATION_BEYOND_MARKOV        | POOLED_SEMIMARKOV_DURATION       | POOLED_MARKOV                  |         40 |            0.0137980 |                   0.0071755 |                   0.0214973 |                   1.0000000 |
| S12F-CANDIDATE-02 | EARLY_MATRIX_BEYOND_POOLED    | EARLY_PREFIX_MATRIX_MARKOV       | POOLED_MARKOV                  |         40 |           -0.0202512 |                  -0.0433061 |                   0.0005670 |                   0.0273438 |
| S12F-CANDIDATE-02 | STABLE_MATRIX_BEYOND_DURATION | EARLY_PREFIX_MATRIX_SEMIMARKOV   | POOLED_SEMIMARKOV_DURATION     |         40 |           -0.0357095 |                  -0.0545300 |                  -0.0182496 |                   0.0000000 |
| S12F-CANDIDATE-02 | CURRENT_UPDATE_BEYOND_EARLY   | CURRENT_PREFIX_MATRIX_SEMIMARKOV | EARLY_PREFIX_MATRIX_SEMIMARKOV |         40 |            0.0197881 |                   0.0105472 |                   0.0303779 |                   1.0000000 |
| S12F-CANDIDATE-03 | DURATION_BEYOND_MARKOV        | POOLED_SEMIMARKOV_DURATION       | POOLED_MARKOV                  |         40 |            0.0158223 |                   0.0087677 |                   0.0234776 |                   1.0000000 |
| S12F-CANDIDATE-03 | EARLY_MATRIX_BEYOND_POOLED    | EARLY_PREFIX_MATRIX_MARKOV       | POOLED_MARKOV                  |         40 |           -0.0248546 |                  -0.0587765 |                   0.0039552 |                   0.0507812 |
| S12F-CANDIDATE-03 | STABLE_MATRIX_BEYOND_DURATION | EARLY_PREFIX_MATRIX_SEMIMARKOV   | POOLED_SEMIMARKOV_DURATION     |         40 |           -0.0429764 |                  -0.0734309 |                  -0.0162200 |                   0.0000000 |
| S12F-CANDIDATE-03 | CURRENT_UPDATE_BEYOND_EARLY   | CURRENT_PREFIX_MATRIX_SEMIMARKOV | EARLY_PREFIX_MATRIX_SEMIMARKOV |         40 |            0.0276696 |                   0.0113935 |                   0.0465730 |                   1.0000000 |

## F12 joint-process probability reconstruction

| candidateId       | modelId                          |   states |   equalMatrixMeanBranchLogLoss |   equalMatrixMeanBranchBrier |     qRmse |   qSpearman |
|:------------------|:---------------------------------|---------:|-------------------------------:|-----------------------------:|----------:|------------:|
| S12F-CANDIDATE-02 | CURRENT_PREFIX_MATRIX_SEMIMARKOV |      200 |                      0.6757746 |                    0.2215367 | 0.2018605 |   0.6708305 |
| S12F-CANDIDATE-02 | EARLY_PREFIX_MATRIX_MARKOV       |      200 |                      0.7305787 |                    0.2427815 | 0.2483354 |   0.5382875 |
| S12F-CANDIDATE-02 | EARLY_PREFIX_MATRIX_SEMIMARKOV   |      200 |                      0.7310815 |                    0.2328793 | 0.2281449 |   0.6139980 |
| S12F-CANDIDATE-02 | POOLED_IID                       |      200 |                      0.7262981 |                    0.2664702 | 0.2896353 | nan         |
| S12F-CANDIDATE-02 | POOLED_MARKOV                    |      200 |                      0.6661536 |                    0.2365735 | 0.2333390 |   0.4174783 |
| S12F-CANDIDATE-02 | POOLED_SEMIMARKOV_DURATION       |      200 |                      0.6330212 |                    0.2209069 | 0.1988499 |   0.5987419 |
| S12F-CANDIDATE-03 | CURRENT_PREFIX_MATRIX_SEMIMARKOV |      200 |                      0.6651714 |                    0.2238796 | 0.2176788 |   0.6150425 |
| S12F-CANDIDATE-03 | EARLY_PREFIX_MATRIX_MARKOV       |      200 |                      0.7580022 |                    0.2551474 | 0.2798000 |   0.4463966 |
| S12F-CANDIDATE-03 | EARLY_PREFIX_MATRIX_SEMIMARKOV   |      200 |                      0.7455796 |                    0.2445284 | 0.2605179 |   0.5092875 |
| S12F-CANDIDATE-03 | POOLED_IID                       |      200 |                      0.7361385 |                    0.2712578 | 0.3050548 | nan         |
| S12F-CANDIDATE-03 | POOLED_MARKOV                    |      200 |                      0.6721878 |                    0.2395833 | 0.2486027 |   0.4145449 |
| S12F-CANDIDATE-03 | POOLED_SEMIMARKOV_DURATION       |      200 |                      0.6306394 |                    0.2199103 | 0.2071810 |   0.5933779 |

The empirical committor is the 64-branch L50 probability, not the single realized primary-lineage future. The latter is retained only as a noisy diagnostic.

## Between-matrix versus within-matrix variation

| candidateId       |   betweenMatrixVariance |   withinMatrixVariance |   betweenMatrixFraction |   betweenMatrixFractionLower95 |   betweenMatrixFractionUpper95 |   withinMatrixGenerationSpearman |
|:------------------|------------------------:|-----------------------:|------------------------:|-------------------------------:|-------------------------------:|---------------------------------:|
| S12F-CANDIDATE-02 |               0.0411023 |              0.0161473 |               0.7179491 |                      0.5659494 |                      0.8115840 |                       -0.0247471 |
| S12F-CANDIDATE-03 |               0.0444551 |              0.0191373 |               0.6990629 |                      0.5352775 |                      0.8136101 |                       -0.1002429 |

The balanced one-way decomposition subtracts registered branch-binomial noise. A large between-matrix fraction supports a stable catalytic-network propensity; a large within-matrix component supports state or episode evolution. The within-matrix generation correlation is descriptive and is not required to be positive: regime switching need not form a universal rising trajectory.

## Scientific gates

| gateId                                      | candidateId       | gateFamily              |     lower95 |    spearman |   permutationP | passed   |
|:--------------------------------------------|:------------------|:------------------------|------------:|------------:|---------------:|:---------|
| DURATION::S12F-CANDIDATE-02                 | S12F-CANDIDATE-02 | TRANSITION_DURATION     |   0.0071755 | nan         |    nan         | True     |
| STABLE_MATRIX::S12F-CANDIDATE-02            | S12F-CANDIDATE-02 | EARLY_MATRIX_PROPENSITY |  -0.0545300 | nan         |    nan         | False    |
| CURRENT_UPDATE::S12F-CANDIDATE-02           | S12F-CANDIDATE-02 | LONGITUDINAL_UPDATE     |   0.0105472 | nan         |    nan         | True     |
| COMMITTOR_RECONSTRUCTION::S12F-CANDIDATE-02 | S12F-CANDIDATE-02 | F12_JOINT_PROCESS       |  -0.0934170 |   0.6708305 |      0.0019493 | False    |
| BETWEEN_MATRIX_DOMINANCE::S12F-CANDIDATE-02 | S12F-CANDIDATE-02 | VARIANCE_DECOMPOSITION  |   0.5659494 | nan         |    nan         | True     |
| DURATION::S12F-CANDIDATE-03                 | S12F-CANDIDATE-03 | TRANSITION_DURATION     |   0.0087677 | nan         |    nan         | True     |
| STABLE_MATRIX::S12F-CANDIDATE-03            | S12F-CANDIDATE-03 | EARLY_MATRIX_PROPENSITY |  -0.0734309 | nan         |    nan         | False    |
| CURRENT_UPDATE::S12F-CANDIDATE-03           | S12F-CANDIDATE-03 | LONGITUDINAL_UPDATE     |   0.0113935 | nan         |    nan         | True     |
| COMMITTOR_RECONSTRUCTION::S12F-CANDIDATE-03 | S12F-CANDIDATE-03 | F12_JOINT_PROCESS       |  -0.0761521 |   0.6150425 |      0.0019493 | False    |
| BETWEEN_MATRIX_DOMINANCE::S12F-CANDIDATE-03 | S12F-CANDIDATE-03 | VARIANCE_DECOMPOSITION  |   0.5352775 | nan         |    nan         | True     |
| COMPLETE_CROSS_CANDIDATE_ADJUDICATION       | BOTH              | COMPLETE                | nan         | nan         |    nan         | True     |

## Interpretation boundary

A favorable result supports a compact stochastic-process description of local compositional heredity under this reconstructed simulator. It does not establish one privileged attractor, an organism, restored molecular identity, independent functional memory, PhiID foresight, author-code identity, intervention efficacy or real chemistry. Matrix prefixes and branch ensembles are simulator-accessible observations. This adaptive analysis reuses L50 outcomes and is not confirmatory.

## Source grounding

The registered hierarchy follows finite-state Markov, renewal/semi-Markov and heterogeneous-chain empirical-Bayes literature. Web research was used only to ground the model family before L51 outcomes; it did not supply or select a favorable parameter. Exact source records and URLs are in `source_registry.parquet`.

## Runtime and provenance

- Repository lock: `0d1c65e9576b8d73b7506c39abe1a708c505b466`.
- Workers: `1`; one numerical-library thread; GPU hours: 0.
- Wall time: `1.542` minutes; CPU upper estimate: `0.025708` hours.
- New matrices, primary trajectories and branch streams: 0, 0 and 0.
- Frozen branch sequences analyzed: `51,200`; transition observations: `614,400`.
- Matrix bootstraps: 4096; whole-matrix permutations: 512.

## Limitations

The process is binary and threshold-defined, even though the threshold is frozen rather than searched. Matrix-prefix estimates use one realized lineage and are shrunk rather than direct latent network parameters. Five landmarks cannot capture every episode. Branches and landmarks within a catalytic matrix are dependent; all uncertainty resamples matrices. Exact post-fission alignment controls within-cycle phase rather than estimating its effect. The result remains exploratory and cannot retroactively change L44, L50 or S18.
