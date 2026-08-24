# S19-L20 — Multiscale Geometry and Topology Before Recurring-Attractor Entry

## Chief/human handoff

- **Step:** `E01-S19-L20-MULTISCALE-GEOMETRY-TOPOLOGY-EARLY-WARNING-v1.0.0`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** `ATTRACTOR_ONSET_TASK_ESTABLISHED`, `MULTISCALE_GEOMETRY_FAMILY_NON_SUPPORT`, `PERSISTENT_TOPOLOGY_NOT_INCREMENTAL`, `INTRINSIC_GEOMETRY_NOT_INCREMENTAL`, `PATH_GEOMETRY_NOT_INCREMENTAL`, `NOT_PROMOTABLE_AS_CONFIRMED`, `POSSIBLE_STABILITY_PROXY`
- **Selected discovery lead:** `NONE`.
- **Validation:** exact L18 task/split replay, immutable-prior validation through L19, ten preregistered fixtures, independent all-unit feature replay, exact suffix invariance, matrix-level repeated CV, 4,096 bootstraps, 512 max-statistic label permutations, temporal/feature controls, regeneration, storage and artifact hashes passed.
- **Recommended next bounded loop:** Advance to an outcome-blind landmark/survival reformulation in L21; fixed persistent-topology, intrinsic-dimension and path-geometry summaries are pruned.

## Frozen question

Do multiscale point-cloud topology, intrinsic dimensionality, or path geometry calculated only from observations 0–63 predict first entry into the frozen recurring-attractor state during observations 64–191 beyond time, exact adjacent H/stability and prefix recurrence geometry?

## Cohort

| candidateId       |   atRisk |   events |   occupancy |   nonEvents |
|:------------------|---------:|---------:|------------:|------------:|
| S12F-CANDIDATE-02 |       53 |       33 |    0.207904 |          20 |
| S12F-CANDIDATE-03 |       54 |       33 |    0.213186 |          21 |

This is the exact L18/L19 discovery task. Its completed-run attractor target remains retrospective and author-ambiguous; every competitive L20 input is prefix-only.

## Methods

L20 froze three nonduplicative families before outcomes: (1) float64 H0 minimum-spanning-tree persistence plus GUDHI H1 Vietoris–Rips persistence on cosine-chord distances; (2) fixed k=5 and k=10 Levina–Bickel intrinsic-dimension and neighbourhood-contraction summaries; and (3) step, displacement, tortuosity, turning and lag-2/4/8 path geometry. Full-prefix values and registered 32/32 contrasts were evaluated with the unchanged `C=1` L2 logistic model and exact L18 splits. GUDHI 3.13.0 was installed from its CPython 3.13 wheel; no GPU was used.

## Results

| candidateId       | modelId                         |    AUROC |    AUPRC |    BRIER |   BALANCED_ACCURACY |
|:------------------|:--------------------------------|---------:|---------:|---------:|--------------------:|
| S12F-CANDIDATE-02 | COMPACT_BASELINE                | 0.419697 | 0.619252 | 0.277271 |            0.509091 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_INTRINSIC_GEOMETRY | 0.377273 | 0.560682 | 0.311135 |            0.37803  |
| S12F-CANDIDATE-02 | COMPACT_PLUS_MULTISCALE         | 0.49697  | 0.630703 | 0.299909 |            0.508333 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_PATH_GEOMETRY      | 0.34697  | 0.553401 | 0.335742 |            0.398485 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_TOPOLOGY           | 0.625758 | 0.704734 | 0.240347 |            0.643939 |
| S12F-CANDIDATE-02 | DUMMY_TRAINING_PRIOR            | 0.426515 | 0.600295 | 0.235112 |            0.5      |
| S12F-CANDIDATE-02 | EXACT_H_STABILITY               | 0.371212 | 0.587305 | 0.305521 |            0.438636 |
| S12F-CANDIDATE-02 | PREFIX_RECURRENCE_GEOMETRY      | 0.431818 | 0.62959  | 0.27831  |            0.403788 |
| S12F-CANDIDATE-03 | COMPACT_BASELINE                | 0.619048 | 0.746198 | 0.237519 |            0.608225 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_INTRINSIC_GEOMETRY | 0.607504 | 0.751558 | 0.257966 |            0.577922 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_MULTISCALE         | 0.640693 | 0.785686 | 0.260203 |            0.538961 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_PATH_GEOMETRY      | 0.68254  | 0.836266 | 0.242863 |            0.538961 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_TOPOLOGY           | 0.603175 | 0.751763 | 0.251824 |            0.530303 |
| S12F-CANDIDATE-03 | DUMMY_TRAINING_PRIOR            | 0.378066 | 0.553953 | 0.238325 |            0.5      |
| S12F-CANDIDATE-03 | EXACT_H_STABILITY               | 0.68254  | 0.792916 | 0.228637 |            0.577922 |
| S12F-CANDIDATE-03 | PREFIX_RECURRENCE_GEOMETRY      | 0.568543 | 0.716961 | 0.248779 |            0.515152 |

## Gate adjudication

| candidateId       | modelId                         |   atRiskMatrices |   events |   nonEvents | taskEstablished   |    auRoc |   auRocBootstrapLower95 |    auPrc |   prevalence |    brier |   dummyBrier |   deltaOverCompact |   deltaOverExactH |   familywisePermutationP |   leaveOneOutPositiveFraction |   temporalPermutationAuRoc | suffixInvariancePassed   | candidateDiscoveryGatePassed   |
|:------------------|:--------------------------------|-----------------:|---------:|------------:|:------------------|---------:|------------------------:|---------:|-------------:|---------:|-------------:|-------------------:|------------------:|-------------------------:|------------------------------:|---------------------------:|:-------------------------|:-------------------------------|
| S12F-CANDIDATE-02 | COMPACT_PLUS_TOPOLOGY           |               53 |       33 |          20 | True              | 0.625758 |                0.457345 | 0.704734 |     0.622642 | 0.240347 |     0.235112 |          0.206061  |        0.254545   |                0.0896686 |                     1         |                   0.531818 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_INTRINSIC_GEOMETRY |               53 |       33 |          20 | True              | 0.377273 |                0.227601 | 0.560682 |     0.622642 | 0.311135 |     0.235112 |         -0.0424242 |        0.00606061 |                0.962963  |                     0         |                   0.24697  | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_PATH_GEOMETRY      |               53 |       33 |          20 | True              | 0.34697  |                0.19621  | 0.553401 |     0.622642 | 0.335742 |     0.235112 |         -0.0727273 |       -0.0242424  |                0.990253  |                     0         |                   0.216667 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_MULTISCALE         |               53 |       33 |          20 | True              | 0.49697  |                0.328604 | 0.630703 |     0.622642 | 0.299909 |     0.235112 |          0.0772727 |        0.125758   |                0.421053  |                     1         |                   0.425758 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_TOPOLOGY           |               54 |       33 |          21 | True              | 0.603175 |                0.448587 | 0.751763 |     0.611111 | 0.251824 |     0.238325 |         -0.015873  |       -0.0793651  |                0.88694   |                     0.0185185 |                   0.698413 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_INTRINSIC_GEOMETRY |               54 |       33 |          21 | True              | 0.607504 |                0.45561  | 0.751558 |     0.611111 | 0.257966 |     0.238325 |         -0.011544  |       -0.0750361  |                0.873294  |                     0.0555556 |                   0.59596  | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_PATH_GEOMETRY      |               54 |       33 |          21 | True              | 0.68254  |                0.523318 | 0.836266 |     0.611111 | 0.242863 |     0.238325 |          0.0634921 |        0          |                0.483431  |                     1         |                   0.603175 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_MULTISCALE         |               54 |       33 |          21 | True              | 0.640693 |                0.488784 | 0.785686 |     0.611111 | 0.260203 |     0.238325 |          0.021645  |       -0.041847   |                0.695906  |                     0.962963  |                   0.627706 | True                     | False                          |

The discovery gate required the same frozen model in both candidates, AUROC at least 0.65 with a bootstrap lower bound above 0.5, AUPRC above prevalence, no Brier loss against the dummy, positive increments over compact and exact-H baselines, max-statistic permutation `p<=0.10`, at least 90% positive leave-one-out increments, a worse temporal-permutation control, and exact suffix invariance. This is not a confirmation gate.

## Interpretation

Persistent topology can reveal multiscale connectivity and cycles that a single recurrence threshold misses; intrinsic dimension and path geometry can reveal concentration or constrained motion before attractor entry. Failure constrains these fixed implementations on this landmark task, not every possible organization signal. Candidate-specific or stability-explained behavior is retained but cannot count as a solution.

No completed trajectory, completed centroid, suffix statistic, molecular-row pseudoreplication, favorable-candidate pooling, or outcome-guided geometric scale entered a prospective input.

## Runtime and provenance

- Repository lock: `d1051d468786aaadb2de85d3bca288f02b0234c2`.
- CPU float64, `8` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `2064.816`; process CPU hours: `0.416032`.
- Source identities and reconstruction choices are in `source_grounding_registry.csv` and `source_grounding_report.md`.

## Autonomous continuation boundary

L20 is frozen. The human authorization permits one next bounded loop without an intermediate Chief handoff through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
