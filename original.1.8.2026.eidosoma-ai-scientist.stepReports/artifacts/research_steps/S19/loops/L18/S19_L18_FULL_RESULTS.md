# S19-L18 — Past-Only Organizational Early Warning Before a Recurring-Attractor Onset

## Chief/human handoff

- **Step:** `E01-S19-L18-RECURRING-ATTRACTOR-ONSET-EARLY-WARNING-v1.0.0`
- **Status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Outcome classification:** `EARLY_WARNING_NOT_SUPPORTED_WITHIN_FROZEN_ATTRACTOR_TASK`, `BGM_PREFIX_NOT_INCREMENTAL`, `ATTRACTOR_ONSET_TASK_ESTABLISHED`, `TARGET_RETROSPECTIVE_AUTHOR_AMBIGUITY_UNRESOLVED`, `NOT_PROMOTABLE`
- **Validation:** exact frozen-target replay, prefix-only feature replay, suffix-invariance, matrix-grouped cross-validation, 4,096 matrix bootstraps, 512 label permutations per registered primary model/candidate, immutable-prior, runtime/storage, regeneration, and artifact hashes passed.
- **Lay summary:** L18 replaced the nearly universal adjacent-H state with a first-entry event into a frozen recurring compositional attractor. This created a real at-risk cohort. Whether past-only organization provided incremental warning is reported below; completed-fit and completed-centroid results remain diagnostic oracles.
- **Recommended next action:** mandatory human review. No L19, S20, E02, author contact, confirmation, intervention, or report bundle is active.

## Frozen question

Among matrices that have not entered the frozen L02 dominant recurring-component state by selected-clock observation 64, can organization measured only from those first 64 observations predict first entry during the next 128 observations, beyond time, exact adjacent H, composition stability, and prefix recurrence geometry?

## Why this task is scientifically different

The S13Y adjacent-incoming label is exactly `Y=I(H>0.9)` and is positive on about 98% of molecular observations, so it provides almost no genuine pre-replicator comparison. L18 uses an already frozen L02 recurring-attractor target whose whole-run occupancy is much lower and treats first entry as an event. The landmark and horizon are fixed raw selected-clock counts (64 and 128), so completed trajectory length, padding, and the unknown future suffix cannot define predictor time.

The target itself is reconstructed from the completed run. It is therefore a retrospective outcome adjudication, not an online author label. All competitive predictors are past-only; the completed BGM and target-centroid models are explicitly excluded future-dependent oracles.

## Cohort geometry

| candidateId       |   atRisk |   events |   eventPrevalence |   meanWholeRunOccupancy |
|:------------------|---------:|---------:|------------------:|------------------------:|
| S12F-CANDIDATE-02 |       53 |       33 |          0.622642 |                0.287156 |
| S12F-CANDIDATE-03 |       54 |       33 |          0.611111 |                0.270883 |

## Frozen predictors and model

The non-Phi predictors comprise a fixed time/mass control, exact adjacent-H and composition-stability summaries, nonadjacent prefix recurrence geometry, and organization dynamics (diversity, effective dimension, contraction, curvature, and directional persistence). L17's BreakingGRNMemories lineage was applied unchanged to the 64-observation prefix for separate emergence and integrated summaries. This prefix application is an exploratory causal companion, not a public source-specified GARD mode.

Every model was a fixed L2 logistic regression (`C=1`, no class weighting) with train-only median imputation, missing indicators, and standardization. Ten repeated five-fold matrix-grouped splits were identical across models and candidates remained separate.

## Primary results

| candidateId       | modelId                          |    AUROC |    AUPRC |    BRIER |   BALANCED_ACCURACY |
|:------------------|:---------------------------------|---------:|---------:|---------:|--------------------:|
| S12F-CANDIDATE-02 | COMPLETED_BGM_ORACLE             | 0.456061 | 0.620783 | 0.278154 |            0.524242 |
| S12F-CANDIDATE-02 | COMPLETED_TARGET_CENTROID_ORACLE | 0.543939 | 0.639396 | 0.246423 |            0.518939 |
| S12F-CANDIDATE-02 | DUMMY_TRAINING_PRIOR             | 0.426515 | 0.600295 | 0.235112 |            0.5      |
| S12F-CANDIDATE-02 | EXACT_H_STABILITY                | 0.371212 | 0.587305 | 0.305521 |            0.438636 |
| S12F-CANDIDATE-02 | PAST_FULL_NO_BGM                 | 0.390909 | 0.560826 | 0.337826 |            0.433333 |
| S12F-CANDIDATE-02 | PAST_FULL_WITH_BGM_EMERGENCE     | 0.44697  | 0.602719 | 0.334885 |            0.443182 |
| S12F-CANDIDATE-02 | PREFIX_RECURRENCE_GEOMETRY       | 0.431818 | 0.62959  | 0.27831  |            0.403788 |
| S12F-CANDIDATE-03 | COMPLETED_BGM_ORACLE             | 0.239538 | 0.475351 | 0.331029 |            0.320346 |
| S12F-CANDIDATE-03 | COMPLETED_TARGET_CENTROID_ORACLE | 0.650794 | 0.708444 | 0.233004 |            0.623377 |
| S12F-CANDIDATE-03 | DUMMY_TRAINING_PRIOR             | 0.378066 | 0.553953 | 0.238325 |            0.5      |
| S12F-CANDIDATE-03 | EXACT_H_STABILITY                | 0.68254  | 0.792916 | 0.228637 |            0.577922 |
| S12F-CANDIDATE-03 | PAST_FULL_NO_BGM                 | 0.683983 | 0.778767 | 0.236935 |            0.664502 |
| S12F-CANDIDATE-03 | PAST_FULL_WITH_BGM_EMERGENCE     | 0.69697  | 0.787872 | 0.236443 |            0.625541 |
| S12F-CANDIDATE-03 | PREFIX_RECURRENCE_GEOMETRY       | 0.568543 | 0.716961 | 0.248779 |            0.515152 |

## Gate adjudication

| candidateId       |   atRiskMatrices |   events |   nonEvents |   meanWholeTrajectoryOccupancy | taskEstablished   |   primaryAuRoc |   primaryAuRocBootstrapLower95 |   primaryAuPrcMinusPrevalenceLower95 |   primaryBrier |   dummyBrier | outperformsAllRegisteredBaselines   | permutationPassed   | suffixInvariancePassed   | pastOnlyLeadGatePassed   | nonPhiOrganizationSignalDescriptive   |
|:------------------|-----------------:|---------:|------------:|-------------------------------:|:------------------|---------------:|-------------------------------:|-------------------------------------:|---------------:|-------------:|:------------------------------------|:--------------------|:-------------------------|:-------------------------|:--------------------------------------|
| S12F-CANDIDATE-02 |               53 |       33 |          20 |                       0.287156 | True              |        0.44697 |                       0.287322 |                           -0.165656  |       0.334885 |     0.235112 | False                               | False               | True                     | False                    | False                                 |
| S12F-CANDIDATE-03 |               54 |       33 |          21 |                       0.270883 | True              |        0.69697 |                       0.541253 |                            0.0341662 |       0.236443 |     0.238325 | False                               | True                | True                     | False                    | True                                  |

The primary lead gate requires, in both candidates, a bootstrap-lower AUROC above 0.5, AUPRC above prevalence, Brier improvement over the training-prior dummy, favorable paired bootstrap differences over time, exact-H/stability, recurrence geometry and the complete non-Phi model, matrix-label permutation rejection, and exact suffix invariance. A future-dependent oracle can never satisfy this gate.

## Controls and validation

- Frozen L02 target labels and centroid scores were recomputed from all 200 immutable trajectories and matched exactly.
- Every past-only feature was unchanged after suffix deletion/shuffle at registered sentinels; prefix BGM arrays replayed exactly.
- Completed-fit BGM prefix values were separately shown to be suffix-sensitive and remained diagnostic only.
- Molecular observations were never treated as independent samples; the catalytic matrix was the unit throughout.
- Within-prefix temporal permutation and 512 matrix-label permutations were retained as negative controls.
- Scientific tables were regenerated deterministically from the frozen feature/result payloads.

## Interpretation boundary

This adaptive L18 task can reveal a useful reaction-coordinate lead, but it cannot identify the paper's unavailable label implementation or prove that causal emergence predicts replication. The target was selected after prior L02 evidence and depends on the completed run. Any positive result requires a new seed-firewalled confirmation in which the target, landmark, horizon, predictors, and gates are frozen before simulation. A null result constrains this particular attractor-onset task without proving that no early organization signal exists.

## Runtime and provenance

- Repository lock: `f31bd46e6df2af184faa37481684bd5a8129aff2`.
- CPU float64, no GPU, one numerical-library thread per worker.
- Wall seconds: `4232.592`; reported worker CPU hours: `0.865665`.
- Source and prior identities are in `source_snapshot_manifest.json` and `immutable_prior_validation.json`.

## Mandatory boundary

Stop here for human review. Do not begin L19, S20, E02, confirmation, intervention, author contact, or report generation automatically.
