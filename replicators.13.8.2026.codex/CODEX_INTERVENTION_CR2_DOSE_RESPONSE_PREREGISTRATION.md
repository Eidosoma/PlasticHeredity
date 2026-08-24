# Codex CR2 graded molecular dose-response confirmation

This document freezes the full CR2 experiment before any CR2 scientific
future is generated. CR2 is permitted because the prospectively frozen full
CR1 confirmation passed all four candidate-by-branch-half cells. CR1 outcomes
are used only for this phase-advancement decision; their effect sizes are not
used to choose CR2 arms, thresholds, seeds, analyses, or gates.

## Scientific question

CR1 established that the two most extreme edits selected by the frozen Codex
predictor causally separate `JOINT_BREAK_RUN3` probability. CR2 tests whether
the predictor contains graded causal information across intermediate legal
edits, rather than merely identifying a useful extreme pair.

## Frozen states and predictor

- Reuse exactly the 200 catalytic matrices, candidates 02 and 03, and natural
  post-fission landmarks 20, 35, 50, 65, and 80 from full CR1.
- Deterministically reconstruct all 2,000 states from CR1's sealed cohort seed
  and label, then compare state IDs, candidate and matrix labels, landmarks,
  compositions, generation and clock values, complete inheritance histories,
  complete boundary-similarity histories, and beta matrices against CR1's
  checksum-sealed `state_and_matrix_arrays.npz` before launching a future.
- Use the byte-identical frozen candidate-separated 5x composite predictor
  copied from the sealed CR1 registration.
- Do not refit, recalibrate, simplify, threshold, or otherwise alter the model.
- CR1 realized outcomes are not read by edit selection or CR2 inference.

## Frozen six-arm selection

For every restored state, score every legal mass-preserving one-molecule
substitution with the frozen predictor. Let `K` be the number of legal edits.
For empirical quantile `q` in `0, 0.2, 0.4, 0.6, 0.8, 1`, select the actual
order statistic at:

```
rank(q) = floor(q * (K - 1) + 0.5)
```

Probabilities are ordered from low to high. If the selected order-statistic
probability is tied, choose the lexicographically first `(remove_type,
add_type)` edit with that exact float64 probability. The frozen arms, in
increasing intended dose order, are:

```
Q00, Q20, Q40, Q60, Q80, Q100
```

The selected probability shift is always recorded relative to the unedited
state's frozen prediction, although NOOP is not a CR2 simulation arm. Persist
every scored legal edit, every selected rank and edit, and all predicted
probabilities. Realized outcomes never enter selection.

## Futures and randomness

- 64 F12 futures per arm and state.
- Fixed halves A = branches 0--31 and B = branches 32--63.
- 2,000 states x 6 arms x 64 branches = 768,000 primary futures.
- Replay all 768,000 futures from the reconstructed states and sealed seeds.
- Use a new CR2 future seed domain disjoint from CR1 and every earlier phase.
- For a candidate, matrix, landmark, and branch, all six arms receive the same
  seed. Arm identity is absent from the future seed key. These are common
  random streams, not necessarily identical realized futures after divergence.
- Quantile selection is deterministic and consumes no future randomness.
- No future retry, matrix replacement, state exclusion, or candidate pooling.

## Primary analyses and gates

Analyze candidate 02 half A, candidate 02 half B, candidate 03 half A, and
candidate 03 half B separately. The catalytic matrix is always the inference
unit; its five states and all associated arms remain together.

For each state and half:

1. Compute the six realized `JOINT_BREAK_RUN3` branch probabilities.
2. Compute Spearman correlation across the six frozen predicted shifts and six
   realized probabilities, using average ranks for ties. If either six-value
   rank vector is constant, assign correlation zero; the state is not dropped.
3. State-center both predicted shifts and realized probabilities across the six
   edits. Fit the through-origin pooled slope from the centered cross-products.

Average state correlations within each matrix, then across matrices. Estimate
the centered slope from sums of matrix-level numerators and denominators. Use
4,096 shared whole-matrix bootstrap draws for both quantities. Also calculate
4,096 whole-matrix sign-randomization p-values, Holm-adjusted across the four
cells separately for the Spearman and slope families; these p-values are
reported diagnostics, not additional gates because the external CR2 directive
defines its gate using estimates and bootstrap intervals.

Each of all four cells must satisfy:

- mean within-state Spearman correlation > 0;
- its 95% matrix-bootstrap lower bound > 0;
- state-centered calibration slope > 0; and
- its 95% matrix-bootstrap lower bound > 0.

The CR2 gate passes only if every condition passes in all four cells and exact
replay, state reconstruction, and artifact readback all pass. Monotonic arm
means, attenuation, adjusted p-values, landmark summaries, matrix effects, and
the count of zero-information state correlations are reported descriptively.
The frozen predictor is not recalibrated even if its slope differs from one.

## Validation, sealing, and stop

Before CR2 scientific futures:

- pass the inherited intervention validation suite plus CR2-specific tests;
- seal this document, source, tests, shared dependency hashes, frozen-model
  hash, CR1 result and state-artifact hashes, endpoint definitions, seed
  registry, inference contract, and artifact contract;
- run only a non-scientific I/O, legality, checkpoint, and replay smoke that
  discloses no effect size, event rate, arm ordering, or candidate difference;
  and
- require at least 18 projected CPU-hours in the new compute allocation.

After the result is checksum-sealed, stop for review. Do not automatically
launch CR5, CR6, CR7, a new predictor search, or any rescue analysis.

## Claim boundary

A pass may support that the frozen Codex predictor ranks small molecular edits
by graded causal influence on the operational simulated `JOINT_BREAK_RUN3`
process. It does not establish strict-eight control, autonomous agency,
biological memory, life, an autonomous attractor, real prebiotic chemistry, or
a universal origin-of-life mechanism.
