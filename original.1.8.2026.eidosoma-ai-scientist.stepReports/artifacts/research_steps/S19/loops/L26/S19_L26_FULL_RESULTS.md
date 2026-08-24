# S19-L26 — Online Recurrence-Map Analog Committor Before Attractor Entry

## Chief/human handoff

- **Step:** `E01-S19-L26-RECURRENCE-MAP-ANALOG-COMMITTOR-v1.0.0`
- **Status:** complete within the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** `RECURRENCE_MAP_ANALOG_NON_SUPPORT`, `ANALOG_COMMITTOR_NOT_INCREMENTAL`, `NOT_PROMOTABLE_AS_CONFIRMED`, `POSSIBLE_STABILITY_PROXY`
- **Selected lead:** `NONE`.
- **Validation:** immutable L25/prior hashes; exact L23 task/cache/firewall replay; development-only normalization and analogue library lock before validation access; candidate-separated metrics; 4,096 matrix bootstraps; 512 development and validation label permutations; temporal-reversal and suffix controls; exact representation/prediction/report regeneration; storage and artifact hashes passed.
- **Recommended next bounded loop:** Do not tune k or the recurrence map; test one transition-tube density/current hypothesis in L27.

## Frozen question and method

At five fixed landmarks, does the complete nonadjacent cosine recurrence map over the previous 64 observations support an analogue estimate of entry during the next 32 observations beyond exact-H traces, ordinary non-H organization paths and elapsed-time priors? A fixed 15-neighbor, uniform-weight library was constructed only from development matrices; neighbors were restricted to the same landmark and candidate identity was not a feature. Validation remained candidate separated. The target is a completed-run recurring-attractor reconstruction, so even a passing predictor would require untouched confirmation and would not identify the paper's author implementation.

## Held-out results

| candidateId       | modelId               |   rows |   events |   prevalence |    AUROC |    AUPRC |    BRIER |
|:------------------|:----------------------|-------:|---------:|-------------:|---------:|---------:|---------:|
| S12F-CANDIDATE-02 | EXACT_H_TRACE_ANALOG  |    300 |       61 |     0.203333 | 0.50415  | 0.203475 | 0.170267 |
| S12F-CANDIDATE-02 | LANDMARK_PRIOR        |    300 |       61 |     0.203333 | 0.5107   | 0.205572 | 0.162211 |
| S12F-CANDIDATE-02 | ORDINARY_PATH_ANALOG  |    300 |       61 |     0.203333 | 0.598429 | 0.252274 | 0.160578 |
| S12F-CANDIDATE-02 | RECURRENCE_MAP_ANALOG |    300 |       61 |     0.203333 | 0.500892 | 0.209466 | 0.170133 |
| S12F-CANDIDATE-03 | EXACT_H_TRACE_ANALOG  |    288 |       60 |     0.208333 | 0.572588 | 0.238633 | 0.166636 |
| S12F-CANDIDATE-03 | LANDMARK_PRIOR        |    288 |       60 |     0.208333 | 0.526243 | 0.222409 | 0.164733 |
| S12F-CANDIDATE-03 | ORDINARY_PATH_ANALOG  |    288 |       60 |     0.208333 | 0.467361 | 0.198914 | 0.178364 |
| S12F-CANDIDATE-03 | RECURRENCE_MAP_ANALOG |    288 |       60 |     0.208333 | 0.542836 | 0.247118 | 0.167793 |

## Landmark diagnostics

| candidateId       |   landmark |   rows |   events |    AUROC |    AUPRC |     BRIER |
|:------------------|-----------:|-------:|---------:|---------:|---------:|----------:|
| S12F-CANDIDATE-02 |         64 |     90 |       18 | 0.370756 | 0.171212 | 0.18      |
| S12F-CANDIDATE-02 |         96 |     72 |       16 | 0.63058  | 0.381149 | 0.167901  |
| S12F-CANDIDATE-02 |        128 |     56 |       10 | 0.601087 | 0.229752 | 0.146667  |
| S12F-CANDIDATE-02 |        160 |     46 |       10 | 0.534722 | 0.272218 | 0.18058   |
| S12F-CANDIDATE-02 |        192 |     36 |        7 | 0.433498 | 0.216883 | 0.173086  |
| S12F-CANDIDATE-03 |         64 |     91 |       21 | 0.465306 | 0.24393  | 0.189548  |
| S12F-CANDIDATE-03 |         96 |     70 |       18 | 0.534188 | 0.272122 | 0.197016  |
| S12F-CANDIDATE-03 |        128 |     52 |       11 | 0.51663  | 0.258824 | 0.171282  |
| S12F-CANDIDATE-03 |        160 |     41 |        7 | 0.676471 | 0.470175 | 0.133333  |
| S12F-CANDIDATE-03 |        192 |     34 |        3 | 0.693548 | 0.185185 | 0.0856209 |

## Gate adjudication

| candidateId       |   primaryAUROC |   exactHAUROC |   ordinaryAUROC |   bootstrapLower |   deltaExactHBootstrapLower |   deltaOrdinaryBootstrapLower |   developmentFamilywiseP |   validationFamilywiseP |   agreeingLandmarks | auRocPointPassed   | auRocBootstrapPassed   | pointOverExactHPassed   | pointOverOrdinaryPassed   | bootstrapOverExactHPassed   | bootstrapOverOrdinaryPassed   | auPrcPassed   | brierPassed   | developmentPermutationPassed   | validationPermutationPassed   | landmarksPassed   | suffixPassed   | candidateDiscoveryGatePassed   |
|:------------------|---------------:|--------------:|----------------:|-----------------:|----------------------------:|------------------------------:|-------------------------:|------------------------:|--------------------:|:-------------------|:-----------------------|:------------------------|:--------------------------|:----------------------------|:------------------------------|:--------------|:--------------|:-------------------------------|:------------------------------|:------------------|:---------------|:-------------------------------|
| S12F-CANDIDATE-02 |       0.500892 |      0.50415  |        0.598429 |         0.418266 |                   -0.108291 |                    -0.198876  |                 0.746589 |                0.732943 |                   3 | False              | False                  | False                   | False                     | False                       | False                         | True          | False         | False                          | False                         | False             | True           | False                          |
| S12F-CANDIDATE-03 |       0.542836 |      0.572588 |        0.467361 |         0.454684 |                   -0.136586 |                    -0.0409815 |                 0.20078  |                0.265107 |                   4 | False              | False                  | False                   | True                      | False                       | False                         | True          | False         | False                          | False                         | True              | True           | False                          |

## Interpretation

The recurrence map retains much more path geometry than L25's local operator summaries but is still past-only and molecule-label invariant. Failure of the preregistered incremental gates constrains this one analogue construction; it does not prove that every transition-path coordinate is absent. Any apparent separation that does not exceed exact-H and ordinary-path controls in both simulator candidates remains an exploratory stability proxy.

## Runtime and provenance

- Repository lock: `d5980662da5c459632404b384090ebf9f1bdb68b`.
- CPU float64, one numerical-library thread, no GPU.
- Wall seconds: `750.065`; process CPU hours: `0.208288`.
- Technical-only amendment 001 corrected inherited DataFrame labels being used as NumPy positions and reused the same frozen neighbor identities for the permutation null. Attempt 001 stopped without releasing a scientific result; the target, representations, neighbors, random streams, statistics, gates, and all authoritative scientific values were unchanged.

## Autonomous continuation boundary

L26 is frozen. The human authorization permits one next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
