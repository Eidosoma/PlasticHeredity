# PX3 post-hoc estimator-support audit and closeout

Date: 2026-08-21

Status: **PX3 closed; registered confirmation failure retained**

This audit was performed only after the fresh PX3 confirmation result was
sealed. It is diagnostic, not preregistered inference, and cannot change the
registered classification. No new catalytic matrix, intervention outcome, or
model fit was generated. The frozen result directory was not modified.

## Frozen result that remains controlling

- Registration:
  `bdb2083071141e23bd05dcc8b7fc2744324f2c69294be60972f20dad151817ec`.
- Frozen result: `results/phir_extension/px3_confirmation`.
- Registered classification: `direct_phi_control_not_confirmed`.
- Coverage: 24 entirely fresh matrices, both candidates, landmarks 20/40/60,
  four arms, 64 F8 futures per state, 36,864 futures and 294,912 observed
  fission transitions.
- Complete exact replay and readback passed. The frozen result checksum file
  has SHA-256
  `67ff8665a2992658034adf1ad4d0f265396d44febcd7ccbb598a53358f6df891`;
  the frozen manifest has SHA-256
  `74a4c2eeda418dd386d68b1fd8e427ccc15105edcbf0f6547efa209b9775177b`.

The registered material full-block PHI_UP-minus-PHI_DOWN contrasts were
opposite to the trained direction in every candidate-by-half cell:

| Candidate | Half | Full-block effect [95% CI] | Inherited-fraction effect [95% CI] |
| --- | --- | ---: | ---: |
| 02 | A | -2.52945 [-3.50043, -1.55924] | -0.02094 [-0.03459, -0.00955] |
| 02 | B | -2.38552 [-3.22887, -1.63829] | -0.02235 [-0.03866, -0.00836] |
| 03 | A | -2.76818 [-3.71090, -1.81389] | -0.03054 [-0.04799, -0.01467] |
| 03 | B | -2.68220 [-3.58953, -1.81765] | -0.02387 [-0.04271, -0.00700] |

Thus the prospective gate failed. The targeted PHI_UP edits also reduced,
rather than increased, inheritance by about two to three percentage points.

## Implementation audit

The reversal was not explained by an arm-label, contrast-sign, model-loading,
candidate-mapping, or replay error.

- PHI_UP was the edit with the largest frozen predicted increase and PHI_DOWN
  the edit with the largest predicted decrease. Mean predicted changes were
  +10.2203 versus -10.8087 in candidate 02 and +9.5227 versus -13.3373 in
  candidate 03.
- Predictions from the serialized frozen models reproduced the fitted values
  to floating-point tolerance.
- Selected edits, candidate contracts, state restoration, common random
  streams, endpoint rows, and the complete replay were consistent.
- The observed reversal was present at landmarks 20, 40, and 60, rather than
  being created by one landmark.
- Development and confirmation beta and state summaries showed no gross
  distribution shift. Most selected confirmation features remained within
  development support.

There was a secondary development warning: candidate 02's out-of-fold signal
was concentrated in the deliberately seeded edit families. Its uniformly
filled development edits did not show a stable within-state relationship.
This makes the development gate less general than an exhaustive-edit claim,
but it is not the principal explanation because the fresh exhaustive selector
does move the 16-branch statistic described below.

## Decisive support audit

Development labeled each edit using 16 branches over F8: 128 explicit
parent-daughter pairs. Confirmation evaluated each registered branch half
using 32 branches over F8: 256 pairs. After seeing the failure, the already
generated confirmation trajectories were deterministically remeasured in
contiguous blocks at three supports. Blocks and landmarks were averaged within
matrix. No new stochastic future or scientific outcome was added: where raw
transition arrays were needed, the sealed seed-defined futures were replayed
and checked against the frozen result before rescoring.

| Branches per score block | Pairs per block | Candidate 02 UP-DOWN | Positive matrices | Candidate 03 UP-DOWN | Positive matrices |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 128 | +8.690129 | 21/24 | +12.537380 | 22/24 |
| 32 | 256 | -2.457484 | 3/24 | -2.725190 | 1/24 |
| 64 | 512 | -0.890950 | 3/24 | -1.087114 | 1/24 |

The 32-branch remeasurement equals the mean of the two frozen branch-half
effects to numerical precision. This verifies the diagnostic against the
sealed analysis. These post-hoc values have no registered confidence gate and
do not rescue PX3.

The estimator scale itself changed sharply. Mean NOOP material full-block
scores were 251.5968 and 248.7942 in the 16-branch development data, but only
28.6692 and 27.7313 in the 32-branch confirmation halves. At 16 branches the
positive contrast was driven mainly by whole-system Gaussian mutual
information (approximately +7.84 and +11.60). At 32 branches the whole-system
components reversed to -4.2640 and -4.4219, producing full-block contrasts of
-2.4575 and -2.7252.

The low-dimensional public nine-atom reading did not show the same large
response. Its mean UP-minus-DOWN contrast was approximately +0.007 and -0.001
at 16-branch support, and +0.00963 and -0.00040 at 32-branch support for
candidates 02 and 03 respectively: near zero and inconsistent.

## Why support changes the full-block statistic

All confirmation windows retained 100 active material coordinates. The
whole-system Gaussian mutual information therefore estimates a joint
covariance over 100 past plus 100 future coordinates.

- With 128 pairs, the centered 200-dimensional joint covariance has rank at
  most 127. At least 73 directions are necessarily unsupported by data.
- With 256 pairs, that covariance can become full rank.
- The implementation makes each covariance invertible with a fixed relative
  ridge (`COVARIANCE_RIDGE = 1e-6`). The log-determinant score is consequently
  dominated by the ridge-supported null directions at 128 pairs and enters a
  different numerical regime at 256 pairs.

This is not merely wider uncertainty at the smaller sample size. It changes
the scale and direction of the measured contrast. The development label and
confirmation endpoint were therefore not support-invariant versions of one
stable full-block quantity.

## Scientific interpretation

What is supported:

- The frozen surrogate learned to select edits that move the **128-pair,
  ridge-regularized material statistic** on fresh confirmation trajectories.
  This is a post-hoc finite-sample result.
- The registered 256-pair confirmation cleanly establishes that this learned
  direction does not transfer to the confirmation statistic.
- PX3 exposes an important estimator-stability failure that future
  high-dimensional information studies must test explicitly.

What is not supported:

- stable direct molecular control of the material full-block measure;
- positive coupling between that measure and plastic heredity;
- rescue of the public nine-atom Phi-r result;
- Phi-r as the physical cause of heredity;
- any inference about consciousness, agency, life, or metaphysics.

The established plastic-heredity results are unchanged. Molecular edits,
the outgoing catalytic-support rule, and occupied-network surgery still
causally alter heredity in their separately registered experiments. PX3 only
closes one proposed information-theoretic interpretation of those effects.

## Closeout and future boundary

PX3 is closed with no rerun, weight change, model rescue, or reinterpretation
of its primary gate. Any future full-block study must be a separately
registered experiment and should require, before scientific confirmation:

1. identical transition support in development and confirmation;
2. a sample-size ladder demonstrating stable scale and sign;
3. dimension-to-sample and covariance-rank audits;
4. a preregistered shrinkage or low-dimensional estimator; and
5. a separate test of coupling to heredity.

PX4 remains the next independent registered phase. Its primary endpoint is
the low-dimensional public nine-atom reading, so this PX3 full-block
covariance failure does not invalidate the PX4 primary question. PX4's
material full-block secondary must be labeled support-sensitive and
descriptive.
