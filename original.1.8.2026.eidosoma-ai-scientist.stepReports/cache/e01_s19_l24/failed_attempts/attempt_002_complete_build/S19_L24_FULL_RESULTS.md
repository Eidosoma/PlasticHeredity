# S19-L24 — Event-Aligned Cross-Candidate Pre-Onset Reaction Coordinate

## Chief/human handoff

- **Step:** `E01-S19-L24-EVENT-ALIGNED-REACTION-COORDINATE-v1.0.0`
- **Status:** complete within the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** `EVENT_ALIGNED_REACTION_COORDINATE_NON_SUPPORT`, `TIME_LOCALIZATION_NOT_SUFFICIENT`, `NOT_PROMOTABLE_AS_CONFIRMED`, `POSSIBLE_STABILITY_PROXY`
- **Selected lead:** `NONE`.
- **Validation:** immutable L23/prior hashes; 800/800 frozen trajectory cache identities; 200/200 outcome-blind shared-matrix firewall; coordinate serialized before validation payload access; candidate-separated matched validation; 4,096 pair bootstraps; 4,096 pair-swap and 512 development-label permutations; exact feature/model/report regeneration; suffix, storage and artifact hashes passed.
- **Recommended next bounded loop:** Event alignment did not recover a common signal. Test one outcome-blind online change-point/operator precursor in L25 without retuning L24.

## Frozen question

Can one sparse development-fitted reaction coordinate distinguish the 32 observations immediately before a first recurring-attractor entry from a non-imminent trajectory at the same absolute molecular time, in a held-out matrix half and both simulator candidates?

## Matched support

| matrixRole   | candidateId       |   pairs |
|:-------------|:------------------|--------:|
| DEVELOPMENT  | S12F-CANDIDATE-02 |      25 |
| DEVELOPMENT  | S12F-CANDIDATE-03 |      26 |
| VALIDATION   | S12F-CANDIDATE-02 |      22 |
| VALIDATION   | S12F-CANDIDATE-03 |      25 |

## Methods

L24 generated no matrix or trajectory. The frozen L23 cohort was split by SHA-256 identity into 200 development and 200 validation matrices, paired across candidates. Within each half, first onsets from 128 through 256 defined event windows `[tau-32,tau)`. Controls were distinct, non-event matrices whose first onset occurred after 256 and at least 96 observations later, sampled without replacement at exactly the same endpoint. A 28-feature molecule-label-permutation-invariant summary was fitted with one development-only scaler and L1 logistic coordinate. The exact-H window, ordinary-composition/stability window, and endpoint-only models were locked controls.

## Held-out results

| candidateId       | modelId                   |   pairCount |    AUROC |    AUPRC |    BRIER |   pairedScoreDifferenceMean |
|:------------------|:--------------------------|------------:|---------:|---------:|---------:|----------------------------:|
| S12F-CANDIDATE-02 | EXACT_H_WINDOW            |          22 | 0.545455 | 0.618083 | 0.243881 |                   0.0268382 |
| S12F-CANDIDATE-02 | ORDINARY_STABILITY_WINDOW |          22 | 0.67562  | 0.671099 | 0.225404 |                   0.143044  |
| S12F-CANDIDATE-02 | REACTION_COORDINATE       |          22 | 0.772727 | 0.790267 | 0.196642 |                   0.231214  |
| S12F-CANDIDATE-02 | TIME_ONLY_ENDPOINT        |          22 | 0.5      | 0.5      | 0.25     |                   0         |
| S12F-CANDIDATE-03 | EXACT_H_WINDOW            |          25 | 0.7024   | 0.675621 | 0.229977 |                   0.0617595 |
| S12F-CANDIDATE-03 | ORDINARY_STABILITY_WINDOW |          25 | 0.648    | 0.648016 | 0.235569 |                   0.123455  |
| S12F-CANDIDATE-03 | REACTION_COORDINATE       |          25 | 0.7008   | 0.682163 | 0.225752 |                   0.178574  |
| S12F-CANDIDATE-03 | TIME_ONLY_ENDPOINT        |          25 | 0.5      | 0.5      | 0.25     |                   0         |

## Gate adjudication

| candidateId       |   pairCount |   reactionAuRoc |   exactHAuRoc |   ordinaryAuRoc |   auRocBootstrapLower95 |   pairedDifferenceLower95 |   deltaExactHLower95 |   deltaOrdinaryLower95 |   pairSwapFamilywiseP |   developmentPermutationFamilywiseP | minimumPairCountPassed   | auRocPointPassed   | auRocBootstrapPassed   | pairedDifferencePassed   | pointImprovementExactHPassed   | pointImprovementOrdinaryPassed   | bootstrapImprovementExactHPassed   | bootstrapImprovementOrdinaryPassed   | pairSwapPermutationPassed   | developmentPermutationPassed   | suffixInvariancePassed   | candidateDiscoveryGatePassed   |
|:------------------|------------:|----------------:|--------------:|----------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|----------------------:|------------------------------------:|:-------------------------|:-------------------|:-----------------------|:-------------------------|:-------------------------------|:---------------------------------|:-----------------------------------|:-------------------------------------|:----------------------------|:-------------------------------|:-------------------------|:-------------------------------|
| S12F-CANDIDATE-02 |          22 |        0.772727 |      0.545455 |         0.67562 |                0.605372 |                 0.0861605 |           -0.0268595 |            -0.00335744 |            0.00170857 |                           0.0136452 | True                     | True               | True                   | True                     | True                           | True                             | False                              | False                                | True                        | True                           | True                     | False                          |
| S12F-CANDIDATE-03 |          25 |        0.7008   |      0.7024   |         0.648   |                0.5814   |                 0.0815255 |           -0.1808    |            -0.0448     |            0.0226995  |                           0.0896686 | True                     | True               | True                   | True                     | False                          | True                             | False                              | False                                | True                        | False                          | True                     | False                          |

## Interpretation

The event time used to align a window is known only after the completed trajectory. Therefore even a passing L24 coordinate would be retrospective discovery and could not support online early warning until frozen at an outcome-blind landmark on new seed-firewalled matrices. Exact-H and ordinary-stability controls prevent a smoothness proxy from being treated as independent organization evidence.

## Runtime and provenance

- Repository lock: `c48da968f8b9734c52deb72278816bf20162fd93`.
- CPU float64, one numerical-library thread, no GPU.
- Wall seconds: `1103.954`; process CPU hours: `0.306602`.
- Frozen L23 trajectory payloads remained in `/cache/e01_s19_l23`; L24 retained compact evidence only.

## Autonomous continuation boundary

L24 is frozen. The human authorization permits the next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
