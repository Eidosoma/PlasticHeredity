# S19-L27 — Online Transition-Tube Density and Current Before Attractor Entry

## Chief/human handoff

- **Step:** `E01-S19-L27-TRANSITION-TUBE-DENSITY-CURRENT-v1.0.0`
- **Status:** complete within the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** `TRANSITION_TUBE_NON_SUPPORT`, `FUNCTIONAL_PATH_NOT_INCREMENTAL`, `NOT_PROMOTABLE_AS_CONFIRMED`, `POSSIBLE_STABILITY_PROXY`
- **Selected lead:** `NONE`.
- **Validation:** immutable L26/prior hashes; exact L23 task/cache/firewall replay; development-only same-landmark tube lock before validation access; candidate-separated metrics; 4,096 matrix bootstraps; 512 development and validation label permutations; temporal-reversal and suffix controls; exact representation/model/report regeneration; storage and artifact hashes passed.
- **Recommended next bounded loop:** Do not tune the tube; test one mechanistic catalytic-network/state susceptibility in L28.

## Frozen question and method

For each at-risk online landmark, the most recent 32 states were represented by all eleven invariant organization-channel levels and first differences. Development-only same-landmark diagonal Gaussian prototypes defined event and non-event transition tubes. The full tube was compared with exact-H-only, ordinary non-H, and landmark-prior controls. Candidate identity was not a predictor; all validation and gates remained candidate separated. The target remains a completed-run recurring-attractor reconstruction.

## Held-out results

| candidateId       | modelId                  |   rows |   events |   prevalence |    AUROC |    AUPRC |    BRIER |
|:------------------|:-------------------------|-------:|---------:|-------------:|---------:|---------:|---------:|
| S12F-CANDIDATE-02 | EXACT_H_TRANSITION_TUBE  |    300 |       61 |     0.203333 | 0.518348 | 0.215718 | 0.193415 |
| S12F-CANDIDATE-02 | FULL_TRANSITION_TUBE     |    300 |       61 |     0.203333 | 0.515262 | 0.209386 | 0.193492 |
| S12F-CANDIDATE-02 | LANDMARK_PRIOR           |    300 |       61 |     0.203333 | 0.509603 | 0.20561  | 0.194424 |
| S12F-CANDIDATE-02 | ORDINARY_TRANSITION_TUBE |    300 |       61 |     0.203333 | 0.51485  | 0.207724 | 0.193859 |
| S12F-CANDIDATE-03 | EXACT_H_TRANSITION_TUBE  |    288 |       60 |     0.208333 | 0.578947 | 0.238051 | 0.188778 |
| S12F-CANDIDATE-03 | FULL_TRANSITION_TUBE     |    288 |       60 |     0.208333 | 0.572588 | 0.253055 | 0.188776 |
| S12F-CANDIDATE-03 | LANDMARK_PRIOR           |    288 |       60 |     0.208333 | 0.560307 | 0.231566 | 0.190094 |
| S12F-CANDIDATE-03 | ORDINARY_TRANSITION_TUBE |    288 |       60 |     0.208333 | 0.564766 | 0.25112  | 0.189036 |

## Landmark diagnostics

| candidateId       |   landmark |   rows |   events |    AUROC |    AUPRC |     BRIER |
|:------------------|-----------:|-------:|---------:|---------:|---------:|----------:|
| S12F-CANDIDATE-02 |         64 |     90 |       18 | 0.489969 | 0.20278  | 0.253442  |
| S12F-CANDIDATE-02 |         96 |     72 |       16 | 0.555804 | 0.315847 | 0.183891  |
| S12F-CANDIDATE-02 |        128 |     56 |       10 | 0.48913  | 0.220362 | 0.15157   |
| S12F-CANDIDATE-02 |        160 |     46 |       10 | 0.541667 | 0.233717 | 0.171846  |
| S12F-CANDIDATE-02 |        192 |     36 |        7 | 0.714286 | 0.318197 | 0.155687  |
| S12F-CANDIDATE-03 |         64 |     91 |       21 | 0.541497 | 0.283347 | 0.250316  |
| S12F-CANDIDATE-03 |         96 |     70 |       18 | 0.455128 | 0.252444 | 0.197183  |
| S12F-CANDIDATE-03 |        128 |     52 |       11 | 0.667406 | 0.338074 | 0.167212  |
| S12F-CANDIDATE-03 |        160 |     41 |        7 | 0.470588 | 0.191026 | 0.148959  |
| S12F-CANDIDATE-03 |        192 |     34 |        3 | 0.526882 | 0.155128 | 0.0877523 |

## Gate adjudication

| candidateId       |   primaryAUROC |   exactHAUROC |   ordinaryAUROC |   bootstrapLower |   deltaExactHBootstrapLower |   deltaOrdinaryBootstrapLower |   developmentFamilywiseP |   validationFamilywiseP |   agreeingLandmarks | auRocPointPassed   | auRocBootstrapPassed   | pointOverExactHPassed   | pointOverOrdinaryPassed   | bootstrapOverExactHPassed   | bootstrapOverOrdinaryPassed   | auPrcPassed   | brierPassed   | developmentPermutationPassed   | validationPermutationPassed   | landmarksPassed   | suffixPassed   | candidateDiscoveryGatePassed   |
|:------------------|---------------:|--------------:|----------------:|-----------------:|----------------------------:|------------------------------:|-------------------------:|------------------------:|--------------------:|:-------------------|:-----------------------|:------------------------|:--------------------------|:----------------------------|:------------------------------|:--------------|:--------------|:-------------------------------|:------------------------------|:------------------|:---------------|:-------------------------------|
| S12F-CANDIDATE-02 |       0.515262 |      0.518348 |        0.51485  |         0.437367 |                  -0.0280914 |                  -0.00814901  |               0.832359   |                1        |                   3 | False              | False                  | False                   | True                      | False                       | False                         | True          | True          | False                          | False                         | False             | True           | False                          |
| S12F-CANDIDATE-03 |       0.572588 |      0.578947 |        0.564766 |         0.492261 |                  -0.0331679 |                  -0.000224494 |               0.00584795 |                0.220273 |                   3 | False              | False                  | False                   | True                      | False                       | False                         | True          | True          | True                           | False                         | False             | True           | False                          |

## Interpretation

This loop tests a functional transition path rather than scalar early-warning summaries, operator changes, or recurrence-map neighbours. A failed common incremental gate constrains this specific source-grounded tube. It does not prove every mechanistic precursor absent, and no discovery result would be confirmatory without new seed-firewalled matrices.

## Runtime and provenance

- Repository lock: `89bf5fb7704ab22d8135be80d3c6499e71cc259b`.
- CPU float64, one numerical-library thread, no GPU.
- Wall seconds: `588.218`; process CPU hours: `0.163294`.

## Autonomous continuation boundary

L27 is frozen. One next bounded loop remains authorized through L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
