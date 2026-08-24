# Frozen-analysis plan summary

## Deterministic rulebook

For each of 200 development and 200 confirmation beta matrices, solve the
fixed points of the normalized expected join-minus-leave composition flow.
Use 16 fixed-seed simplex starts, a 1e-11 update tolerance, and merge forms at
cosine >= 0.95. No outcome is read while deriving forms.

For every retained state, record its nearest form, cosine and L1 distance,
state/form self-support, local flow magnitude and direction, one-step alignment
gain, tangent-space stability, and form diversity/concentration.

## Cross-fitted holding

For a target candidate, state, and branch half, estimate beta-matrix holding
from the opposite candidate, all five landmarks, and the opposite 64-branch
half. Fit binomial ridge models on development matrices only and score
confirmation once. Compare H+C+S with additions of deterministic rulebook
features and cross-candidate holding features.

The primary modeled outcomes are coherence conditional on a run of eight and
the full strict-8 event. Direct matrix-level correlations are also reported for
all four gates and strict-8.

## Fresh causal intervention

For each confirmation state, use seven arms: no-op; toward/away at doses one
and four; and matched random transfers at doses one and four. Every transfer
moves one molecule between already occupied types, preserving mass and
richness. Doses are nested. All arms use the same 64 future random streams and
32-fission horizon.

Primary registered-cosine effects are:

- away minus toward for a break within the first eight fissions;
- toward minus away for eight uninterrupted inherited fissions;
- toward minus away for uninterrupted inheritance plus mutual coherence.

The standard strict-8 net effect is direction-agnostic and secondary because
strict-8 requires both destabilization and subsequent holding.

## Reporting

All inference is grouped by catalytic matrix. Candidate and branch halves are
reported independently. Results remain post hoc and internal unless the
manuscript is separately edited with that status made explicit.

