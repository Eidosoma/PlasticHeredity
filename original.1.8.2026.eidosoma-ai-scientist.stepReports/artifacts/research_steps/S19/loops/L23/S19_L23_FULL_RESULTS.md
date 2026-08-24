# S19-L23 — Powered Independent Screen of Frozen Prefix Organization Families

## Chief/human handoff

- **Step:** `E01-S19-L23-POWERED-FROZEN-PREFIX-FAMILY-SCREEN-v1.0.0`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** `POWERED_ATTRACTOR_ONSET_TASK_ESTABLISHED`, `POWERED_FROZEN_FAMILY_NON_SUPPORT`, `CANDIDATE_HETEROGENEITY_PERSISTS`, `NOT_PROMOTABLE_AS_CONFIRMED`
- **Selected discovery lead:** `NONE`.
- **Validation:** pre-outcome seed/input firewall; 400 shared matrices and 800 registered trajectories without replacement; exact target, trajectory, seed, feature and model replay; candidate-separated matrix CV; 4,096 bootstraps; 512 max-statistic permutations; temporal/feature/suffix controls; immutable-prior, storage and artifact hashes passed.
- **Recommended next bounded loop:** The larger independent cohort rules out simple underpowering of every frozen family. Advance to a compact cross-candidate reaction-coordinate loop rather than retuning these families.

## Frozen question

Does increased independent matrix support reveal a common pre-onset signal in any complete feature family already frozen in L19, L20 or L22?

## Cohort

| candidateId       |   atRisk |   events |   occupancy |   nonEvents |
|:------------------|---------:|---------:|------------:|------------:|
| S12F-CANDIDATE-02 |      182 |      112 |    0.227124 |          70 |
| S12F-CANDIDATE-03 |      181 |      107 |    0.235606 |          74 |

## Methods

Exactly 400 new catalytic matrices and matched initial states were generated before label analysis from a domain-separated root with no detected prior hash or seed-material overlap. Both S13Y simulator candidates completed their registered attempt; incomplete/extinct units were retained and never replaced. The retrospective L02 recurring-attractor label served only as the outcome. All L19 critical-slowing/RQA/DMD, L20 topology/intrinsic/path, and L22 random-convolution implementations were reused unchanged. Six complete registered bundles were tested under one max-statistic family with the exact C=1 logistic estimator.

## Results

| candidateId       | modelId                             |    AUROC |    AUPRC |    BRIER |   BALANCED_ACCURACY |
|:------------------|:------------------------------------|---------:|---------:|---------:|--------------------:|
| S12F-CANDIDATE-02 | COMPACT_BASELINE                    | 0.617602 | 0.685122 | 0.233511 |            0.558036 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L19_ALL                | 0.559566 | 0.688573 | 0.271327 |            0.540179 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_INTRINSIC          | 0.623852 | 0.688137 | 0.235254 |            0.638393 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_MULTISCALE         | 0.641199 | 0.72163  | 0.242266 |            0.621429 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_PATH               | 0.644133 | 0.733714 | 0.231179 |            0.577679 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_TOPOLOGY           | 0.613903 | 0.715829 | 0.238541 |            0.588393 |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L22_RANDOM_CONVOLUTION | 0.486352 | 0.585783 | 0.329612 |            0.489286 |
| S12F-CANDIDATE-02 | DUMMY_TRAINING_PRIOR                | 0.480612 | 0.606365 | 0.2367   |            0.5      |
| S12F-CANDIDATE-02 | EXACT_H_STABILITY                   | 0.630995 | 0.714489 | 0.232104 |            0.566071 |
| S12F-CANDIDATE-03 | COMPACT_BASELINE                    | 0.569841 | 0.62885  | 0.247128 |            0.549507 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L19_ALL                | 0.605077 | 0.725414 | 0.263711 |            0.525638 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_INTRINSIC          | 0.557338 | 0.621224 | 0.256704 |            0.557843 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_MULTISCALE         | 0.596363 | 0.669118 | 0.260785 |            0.582281 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_PATH               | 0.591942 | 0.649772 | 0.249347 |            0.586954 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_TOPOLOGY           | 0.580071 | 0.651792 | 0.251767 |            0.556264 |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L22_RANDOM_CONVOLUTION | 0.59952  | 0.686087 | 0.290001 |            0.566178 |
| S12F-CANDIDATE-03 | DUMMY_TRAINING_PRIOR                | 0.464322 | 0.583067 | 0.241749 |            0.5      |
| S12F-CANDIDATE-03 | EXACT_H_STABILITY                   | 0.526396 | 0.620465 | 0.253952 |            0.51364  |

## Gate adjudication

| candidateId       | modelId                             |   atRiskMatrices |   events |   nonEvents | taskEstablished   |    auRoc |   auRocBootstrapLower95 |    auPrc |   prevalence |    brier |   dummyBrier |   deltaOverCompact |   deltaOverExactH |   familywisePermutationP |   leaveOneOutPositiveFraction |   temporalPermutationAuRoc | suffixInvariancePassed   | candidateDiscoveryGatePassed   |
|:------------------|:------------------------------------|-----------------:|---------:|------------:|:------------------|---------:|------------------------:|---------:|-------------:|---------:|-------------:|-------------------:|------------------:|-------------------------:|------------------------------:|---------------------------:|:-------------------------|:-------------------------------|
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_TOPOLOGY           |              182 |      112 |          70 | True              | 0.613903 |                0.528862 | 0.715829 |     0.615385 | 0.238541 |     0.2367   |        -0.00369898 |       -0.0170918  |                 0.922027 |                     0.0604396 |                   0.625383 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_INTRINSIC          |              182 |      112 |          70 | True              | 0.623852 |                0.538335 | 0.688137 |     0.615385 | 0.235254 |     0.2367   |         0.00625    |       -0.00714286 |                 0.875244 |                     1         |                   0.645663 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_PATH               |              182 |      112 |          70 | True              | 0.644133 |                0.561371 | 0.733714 |     0.615385 | 0.231179 |     0.2367   |         0.0265306  |        0.0131378  |                 0.746589 |                     1         |                   0.66199  | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L19_ALL                |              182 |      112 |          70 | True              | 0.559566 |                0.476074 | 0.688573 |     0.615385 | 0.271327 |     0.2367   |        -0.0580357  |       -0.0714286  |                 1        |                     0         |                   0.554974 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L22_RANDOM_CONVOLUTION |              182 |      112 |          70 | True              | 0.486352 |                0.396553 | 0.585783 |     0.615385 | 0.329612 |     0.2367   |        -0.13125    |       -0.144643   |                 1        |                     0         |                   0.534949 | True                     | False                          |
| S12F-CANDIDATE-02 | COMPACT_PLUS_L20_MULTISCALE         |              182 |      112 |          70 | True              | 0.641199 |                0.558494 | 0.72163  |     0.615385 | 0.242266 |     0.2367   |         0.0235969  |        0.0102041  |                 0.77193  |                     1         |                   0.665944 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_TOPOLOGY           |              181 |      107 |          74 | True              | 0.580071 |                0.495519 | 0.651792 |     0.59116  | 0.251767 |     0.241749 |         0.0102299  |        0.0536752  |                 0.88499  |                     1         |                   0.518439 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_INTRINSIC          |              181 |      107 |          74 | True              | 0.557338 |                0.469154 | 0.621224 |     0.59116  | 0.256704 |     0.241749 |        -0.0125032  |        0.0309422  |                 0.947368 |                     0         |                   0.435716 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_PATH               |              181 |      107 |          74 | True              | 0.591942 |                0.502832 | 0.649772 |     0.59116  | 0.249347 |     0.241749 |         0.0221015  |        0.0655469  |                 0.79922  |                     1         |                   0.464764 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L19_ALL                |              181 |      107 |          74 | True              | 0.605077 |                0.522009 | 0.725414 |     0.59116  | 0.263711 |     0.241749 |         0.0352362  |        0.0786815  |                 0.695906 |                     1         |                   0.443294 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L22_RANDOM_CONVOLUTION |              181 |      107 |          74 | True              | 0.59952  |                0.514936 | 0.686087 |     0.59116  | 0.290001 |     0.241749 |         0.0296792  |        0.0731245  |                 0.744639 |                     1         |                   0.571356 | True                     | False                          |
| S12F-CANDIDATE-03 | COMPACT_PLUS_L20_MULTISCALE         |              181 |      107 |          74 | True              | 0.596363 |                0.507827 | 0.669118 |     0.59116  | 0.260785 |     0.241749 |         0.0265218  |        0.0699672  |                 0.766082 |                     1         |                   0.432432 | True                     | False                          |

In addition to the L19 discovery gates, L23 required at least 150 at-risk matrices and at least 50 events and 50 non-events per candidate. The same frozen family had to pass both candidates. A studied-cohort pass would remain discovery evidence and require another untouched confirmation.

## Interpretation

L23 changes power, not method. It therefore distinguishes small-cohort instability from reproducible signal without creating more opportunities through feature retuning. The target remains a completed-run reconstruction and does not identify author code.

## Runtime and provenance

- Repository lock: `847cb89847707be8bc4b427e577096c01e6d9530`.
- CPU float64, `8` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `2441.031`; process CPU hours: `0.458140`.
- Temporary trajectory payloads remain under `/cache/e01_s19_l23`; compact identities and regeneration evidence are retained in the artifact bundle.

## Autonomous continuation boundary

L23 is frozen. The existing authorization permits one next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
