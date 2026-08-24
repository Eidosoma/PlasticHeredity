# S19-L25 — Online Local-Operator Change Before Recurring-Attractor Entry

## Chief/human handoff

- **Step:** `E01-S19-L25-ONLINE-OPERATOR-CHANGE-PRECURSOR-v1.0.0`
- **Status:** complete within the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** `ONLINE_OPERATOR_CHANGE_NON_SUPPORT`, `LOCAL_TRANSITION_OPERATOR_NOT_INCREMENTAL`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Selected lead:** `NONE`.
- **Validation:** immutable L24/prior hashes; exact L23 cache/task/firewall replay; development-only coordinate lock before validation feature access; candidate-separated online metrics; 4,096 matrix bootstraps; 512 development and validation outcome permutations; suffix and negative controls; exact feature/model/report regeneration; storage and artifact hashes passed.
- **Recommended next bounded loop:** Do not retune the operator features; test one transition-path/committor geometry hypothesis in L26.

## Frozen question

Does a change in local transition dynamics over the previous 64 selected-clock observations predict first recurring-attractor entry in the next 32 observations beyond elapsed time, exact-H changes and all ordinary organization-channel shifts?

## Methods

At fixed landmarks 64, 96, 128, 160 and 192, each still-at-risk matrix contributed one online row. The previous 64 observations were split into 32-reference and 32-recent halves. Eleven molecule-label-invariant organization channels yielded fixed mean, variance and AR(1) shifts plus energy-distance, covariance, ridge transition-operator and speed-change features. One common sparse coordinate was fitted only on the 200 development matrices with equal candidate and matrix weight and was frozen before opening validation features. The target is still a completed-run reconstruction; predictors are strictly past-only.

## Held-out results

| candidateId       | modelId                 |   rows |   events |   prevalence |    AUROC |    AUPRC |    BRIER |
|:------------------|:------------------------|-------:|---------:|-------------:|---------:|---------:|---------:|
| S12F-CANDIDATE-02 | DUMMY_TRAINING_PRIOR    |    300 |       61 |     0.203333 | 0.5      | 0.203333 | 0.192491 |
| S12F-CANDIDATE-02 | EXACT_H_CHANGE          |    300 |       61 |     0.203333 | 0.488442 | 0.201283 | 0.199636 |
| S12F-CANDIDATE-02 | OPERATOR_CHANGE         |    300 |       61 |     0.203333 | 0.52754  | 0.229991 | 0.196613 |
| S12F-CANDIDATE-02 | ORDINARY_CHANNEL_CHANGE |    300 |       61 |     0.203333 | 0.50854  | 0.223282 | 0.197465 |
| S12F-CANDIDATE-02 | TIME_ONLY               |    300 |       61 |     0.203333 | 0.502744 | 0.204019 | 0.195423 |
| S12F-CANDIDATE-03 | DUMMY_TRAINING_PRIOR    |    288 |       60 |     0.208333 | 0.5      | 0.208333 | 0.193711 |
| S12F-CANDIDATE-03 | EXACT_H_CHANGE          |    288 |       60 |     0.208333 | 0.570029 | 0.24943  | 0.191005 |
| S12F-CANDIDATE-03 | OPERATOR_CHANGE         |    288 |       60 |     0.208333 | 0.587646 | 0.263794 | 0.190354 |
| S12F-CANDIDATE-03 | ORDINARY_CHANNEL_CHANGE |    288 |       60 |     0.208333 | 0.572295 | 0.247244 | 0.19194  |
| S12F-CANDIDATE-03 | TIME_ONLY               |    288 |       60 |     0.208333 | 0.566667 | 0.233074 | 0.189139 |

## Landmark diagnostics

| candidateId       |   landmark |   rows |   events |    AUROC |    AUPRC |    BRIER |
|:------------------|-----------:|-------:|---------:|---------:|---------:|---------:|
| S12F-CANDIDATE-02 |         64 |     90 |       18 | 0.515432 | 0.238212 | 0.256939 |
| S12F-CANDIDATE-02 |         96 |     72 |       16 | 0.489955 | 0.310485 | 0.190874 |
| S12F-CANDIDATE-02 |        128 |     56 |       10 | 0.484783 | 0.234359 | 0.164916 |
| S12F-CANDIDATE-02 |        160 |     46 |       10 | 0.638889 | 0.340086 | 0.162722 |
| S12F-CANDIDATE-02 |        192 |     36 |        7 | 0.674877 | 0.329031 | 0.149884 |
| S12F-CANDIDATE-03 |         64 |     91 |       21 | 0.52517  | 0.273277 | 0.251407 |
| S12F-CANDIDATE-03 |         96 |     70 |       18 | 0.732906 | 0.472024 | 0.178146 |
| S12F-CANDIDATE-03 |        128 |     52 |       11 | 0.392461 | 0.182601 | 0.194295 |
| S12F-CANDIDATE-03 |        160 |     41 |        7 | 0.571429 | 0.222669 | 0.144642 |
| S12F-CANDIDATE-03 |        192 |     34 |        3 | 0.483871 | 0.176239 | 0.101176 |

## Gate adjudication

| candidateId       |   validationRows |   validationMatrices |   events |   prevalence |   operatorAuRoc |   exactHAuRoc |   ordinaryAuRoc |   auRocBootstrapLower95 |   deltaExactHLower95 |   deltaOrdinaryLower95 |   developmentPermutationFamilywiseP |   validationPermutationFamilywiseP |   agreeingLandmarks | minimumEventsPassed   | auRocPointPassed   | auRocBootstrapPassed   | pointOverExactHPassed   | pointOverOrdinaryPassed   | bootstrapOverExactHPassed   | bootstrapOverOrdinaryPassed   | auPrcPassed   | brierPassed   | developmentPermutationPassed   | validationPermutationPassed   | landmarkAgreementPassed   | suffixInvariancePassed   | candidateDiscoveryGatePassed   |
|:------------------|-----------------:|---------------------:|---------:|-------------:|----------------:|--------------:|----------------:|------------------------:|---------------------:|-----------------------:|------------------------------------:|-----------------------------------:|--------------------:|:----------------------|:-------------------|:-----------------------|:------------------------|:--------------------------|:----------------------------|:------------------------------|:--------------|:--------------|:-------------------------------|:------------------------------|:--------------------------|:-------------------------|:-------------------------------|
| S12F-CANDIDATE-02 |              300 |                   90 |       61 |     0.203333 |        0.52754  |      0.488442 |        0.50854  |                0.449537 |          0.000383193 |             0.00169132 |                           0.444444  |                          0.783626  |                   3 | True                  | False              | False                  | True                    | True                      | True                        | True                          | True          | True          | False                          | False                         | False                     | True                     | False                          |
| S12F-CANDIDATE-03 |              288 |                   91 |       60 |     0.208333 |        0.587646 |      0.570029 |        0.572295 |                0.508357 |         -0.0246601   |            -0.00417438 |                           0.0272904 |                          0.0604288 |                   3 | True                  | False              | True                   | True                    | True                      | False                       | False                         | True          | True          | True                           | False                         | False                     | True                     | False                          |

## Interpretation

A passing discovery result would still require new seed-firewalled confirmation because the target and L23 cohort have been studied. A failed incremental gate means local operator/change-point statistics do not add reproducible warning information beyond simpler stability summaries under this frozen task; it does not prove that organization has no precursor under every possible definition.

## Runtime and provenance

- Repository lock: `3f4ed0b264094040c45da9db1558fd6160f96785`.
- CPU float64, one numerical-library thread, no GPU.
- Wall seconds: `742.429`; process CPU hours: `0.206189`.

## Autonomous continuation boundary

L25 is frozen. The human authorization permits one next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
