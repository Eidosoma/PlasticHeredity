# Codex full CR4 fixed-composition catalytic-network surgery confirmation

This document prospectively freezes the full CR4 confirmation for the Codex
`JOINT_BREAK_RUN3` intervention program.  The full CR3 outgoing physical-rule
confirmation has passed and is used only as the serial phase-advancement gate.
No CR4 outcome may change this design.

The earlier P3, P3b, P3c, P4a, and shared-break P4 studies remain unchanged.
They are developmental or additive predecessor evidence.  In particular, the
failed P3c omnibus gate is not rescued or relabelled by CR4.

## Scientific questions

At an identical restored composition and identical observed history:

1. Is changing the strength of the catalytic web among currently present
   molecular types sufficient to change F12 break-and-renewal probability?
2. Is a norm-matched surgery placed without reference to the occupied set
   equivalent to no surgery?
3. Does rearranging the occupied catalytic web while preserving its initial
   scalar throughput have a reproducible directional effect, a negligible
   effect, or an unresolved effect?

The strength test and topology classification are separate inferential
families.  Topology is a treatment and is not required to be null.

## Frozen cohort

- 200 completely fresh catalytic matrices shared across candidates 02 and 03;
- untreated natural landmarks 20, 35, 50, 65, and 80;
- 2,000 restored post-fission states with no risk- or outcome-based selection;
- 64 F12 futures per arm and state;
- branch half A = 0--31 and half B = 32--63;
- 640,000 primary futures and a complete 640,000-future replay; and
- no matrix replacement, state replacement, or intervention-future retry.

The endpoint is the existing float64 `JOINT_BREAK_RUN3` definition: within 12
fissions, at least one strict inheritance break (`H <= 0.9`) followed strictly
later by three consecutive inherited boundaries (`H > 0.9`).

## Frozen surgery arms

For current composition `x` and present set `P = {i: x_i > 0}`, Codex stores
`beta[target, catalyst]` and has launch-state catalytic throughput
`T = x.T @ beta @ x`.

1. `LOOSEN`: divide every edge in `beta[P,P]` by 1.5.
2. `TIGHTEN`: multiply every edge in `beta[P,P]` by 1.5.
3. `GLOBAL_RANDOM_SURGERY`: select exactly `|P|^2` distinct beta entries
   uniformly from the complete matrix, without using the identities in `P`;
   apply a balanced log perturbation that preserves positivity and has exact
   Frobenius norm `0.5 * ||beta[P,P]||_F`.
4. `THROUGHPUT_NEUTRAL_TOPOLOGY`: change all and only `beta[P,P]`, preserve
   positivity, use exact Frobenius norm `0.5 * ||beta[P,P]||_F`, and preserve
   launch-state `x.T @ beta @ x` to registered float64 tolerance.
5. `NOOP`: unchanged beta.

The `1.5` and `1/1.5` arms reproduce the corrected external Fable contract.
They are symmetric in log space but not in Frobenius distance: `TIGHTEN` moves
by `0.5 ||beta[P,P]||_F`, while `LOOSEN` moves by
`(1/3) ||beta[P,P]||_F`.  The global and topology controls are matched to the
larger `TIGHTEN` distance.  This asymmetry is explicit and is not repaired
after outcomes.

States with fewer than two occupied types remain in the cohort and use an
all-arm structural no-op.  Every changed edge, achieved norm, selected
location, positivity check, and throughput change is persisted.

## Randomness and replay

CR4 uses new purpose-separated domains for matrix generation, global-random
selection, topology selection, futures, bootstrap, randomization, smoke, and
replay.  Arm identity is absent from each future seed.  Paired arms therefore
receive common random streams, not guaranteed identical realised futures once
their paths diverge.  Selection streams never consume future randomness.

The complete campaign is regenerated from its frozen seeds.  State, surgery,
endpoint, and process digests must match exactly.

## Frozen inference

The catalytic matrix is the inference unit.  All states, landmarks, branch
halves, arms, and candidate observations belonging to one matrix remain
together in each of 4,096 whole-matrix bootstrap draws and 4,096 paired
whole-matrix sign randomizations.  Holm correction covers the four
candidate-by-branch-half cells.

### Primary catalytic-strength gate

CR4 passes only if candidate 02 half A, candidate 02 half B, candidate 03 half
A, and candidate 03 half B each satisfy:

1. `q_LOOSEN - q_TIGHTEN > 0`;
2. the 95% whole-matrix bootstrap lower bound is greater than zero;
3. the Holm-adjusted one-sided randomization `p < 0.05`;
4. `GLOBAL_RANDOM_SURGERY - NOOP` is TOST-equivalent within `+/-0.025`,
   implemented as a 90% whole-matrix bootstrap interval strictly inside that
   margin; and
5. the surgery audit, complete replay, and written-artifact readback pass.

`LOOSEN > NOOP`, `NOOP > TIGHTEN`, and the random-to-target ratio are reported
descriptively but are not CR4 gates.

### Fixed-throughput topology classification

For `THROUGHPUT_NEUTRAL_TOPOLOGY - NOOP`, use a separate two-sided Holm family
across the four cells:

- `reproducible_directional_topology_effect` if all four estimates have the
  same nonzero sign, all four 95% intervals exclude zero, and all four
  Holm-adjusted two-sided randomization p-values are below 0.05;
- `negligible_within_0.025` if every 90% interval lies strictly inside
  `[-0.025,+0.025]`; or
- `inconclusive` otherwise.

This classification cannot invalidate or rescue the primary strength gate.
The previous P3c result makes the directional classification plausible, but
its sign and magnitude do not select any CR4 seed, state, threshold, or gate.

Registered secondary outcomes are break within F12, renewal after the first
break, inherited-boundary count, first-break and certification times,
survival, growth updates, final entropy, and occupied types.  Effects by
landmark and the frozen predictor's predicted versus realised arm shifts are
descriptive.

## Validation, budget, and stop

Before a scientific CR4 matrix exists:

- pass the inherited intervention suite and CR4-specific surgery, stream,
  endpoint, matrix-block, replay, and inference tests;
- verify and hash the sealed CR3 result and all predecessor artifacts that
  motivated the controls;
- seal this document, source closure, tests, dependencies, frozen predictor,
  seed registry, protocol, and artifact contract; and
- pass a non-scientific smoke that discloses no effect size, event rate, arm
  ordering, or candidate difference.

Estimated cost is 14--17 CPU-hours and 1.5--2.0 GB of temporary disk.  Launch
requires at least 20 declared CPU-hours and 3.0 GB free.  Once launched it is
completed, replayed, and sealed rather than killed because an estimate was
imperfect.

After sealing, stop for review.  Do not automatically launch CR5, CR6, CR7,
feedback, transfer, or a new surgery search.

## Claim boundary

A primary pass may support that changing occupied catalytic-network strength
at fixed composition causally changes simulated `JOINT_BREAK_RUN3` probability
under both Codex contracts.  The topology family may separately support that
network arrangement contains causal information not captured by the single
launch-state throughput scalar.

It cannot establish strict-eight control, life, agency, biological memory,
error correction, autonomous organization, an autonomous attractor, real
prebiotic chemistry, Phi/PhiID intervention, or a universal origin-of-life
mechanism.
