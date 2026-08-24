# Codex full CR3 physical-rule confirmation

This document freezes the full prospective confirmation of the externally
specified physical catalytic rule for the operational Codex
`JOINT_BREAK_RUN3` process. The earlier corrected P2b experiment remains a
40-matrix developmental pilot. Its outcomes justify running the already
specified full phase but do not select any CR3 matrix, state, seed, arm,
threshold, analysis, or gate.

## Scientific question

Does a one-molecule substitution chosen only from the outgoing catalytic
influence of each molecular type causally change break-and-renewal probability
in both independently reconstructed Codex simulator candidates?

Codex stores `beta[target, catalyst]` and uses `beta @ n` in its kinetic boost
equation. The frozen physical-rule quantity is therefore

```text
x = n / sum(n)
outgoing = x @ beta = beta.T @ x
```

It is not the incoming quantity `beta @ x` tested by the preserved P2 negative
control.

## Frozen cohort and arms

- 200 completely fresh catalytic matrices shared across candidates 02 and 03;
- untreated natural landmarks 20, 35, 50, 65, and 80;
- 2,000 restored states, with no outcome- or risk-based state selection;
- 64 F12 futures per arm and state;
- fixed halves A = branches 0--31 and B = branches 32--63;
- 512,000 primary futures and a complete 512,000-future replay; and
- no matrix replacement, intervention-future retry, candidate pooling, or
  post-outcome protocol change.

For every state, enumerate every legal mass-preserving one-molecule
substitution. Ties are resolved by the enumeration's fixed lexicographic
order.

- `RULE_DOWN` (stabilizing): remove the present type with the smallest outgoing
  influence and add the distinct type with the largest outgoing influence.
- `RULE_UP` (destabilizing): remove the present type with the largest outgoing
  influence and add the distinct type with the smallest outgoing influence.
- `RANDOM`: choose uniformly from every legal substitution using an independent
  selection stream.
- `NOOP`: leave the state unchanged.

The rule is equivalent to selecting the legal edit that maximizes or minimizes
`outgoing[add] - outgoing[remove]`; this handles present-type and distinct-type
constraints exactly.

## Randomness and replay

The cohort, random-arm selection, future simulation, bootstrap,
randomization, smoke, and replay purposes use separately derived CR3 domains
that are disjoint from every earlier intervention domain. Arm identity is
absent from a branch's future seed, so paired arms receive common random
streams. Random edit selection never consumes the future stream.

The complete primary campaign is repeated from the restored states and frozen
seeds. State, edit, endpoint, and process digests must agree exactly.

## Primary inference and gate

The catalytic matrix is the inference unit. All five landmarks, arms, branch
halves, and both candidate observations associated with a matrix remain
together in each of 4,096 whole-matrix bootstrap draws and 4,096 paired
whole-matrix sign randomizations. Holm adjustment covers the four
candidate-by-half primary cells.

CR3 passes only if candidate 02 half A, candidate 02 half B, candidate 03 half
A, and candidate 03 half B each satisfy:

1. mean paired `RULE_UP - RULE_DOWN > 0`;
2. its 95% whole-matrix bootstrap lower bound is greater than zero;
3. its Holm-adjusted whole-matrix randomization `p < 0.05`;
4. `RANDOM - NOOP` is TOST-equivalent within `+/-0.025`, implemented as a 90%
   whole-matrix bootstrap interval lying strictly inside that margin; and
5. exact replay and artifact readback pass.

The CR1-only `RULE_UP > NOOP`, `NOOP > RULE_DOWN`, and random-to-target-effect
ratio checks are reported descriptively but are not CR3 gates. They are not in
the externally registered CR3 contract.

Registered secondary outcomes are break within F12, run3 after the first
break, inherited-boundary count, first-break time, renewal-certification time,
survival, growth updates, final entropy, and final occupied types. Effects by
landmark, per-matrix effects, expected-sign counts, maximum single-matrix
influence, and the descriptive ratio to the sealed CR1 model-guided effect are
reported.

## Validation, sealing, and operational boundary

Before any CR3 scientific matrix exists:

- pass the inherited intervention validation suite and CR3-specific tests;
- verify the corrected outgoing orientation, legal extrema, tie behavior,
  stream separation, no-op behavior, endpoint fixtures, matrix blocking, and
  replay machinery;
- seal this document, implementation, tests, dependency hashes, frozen model,
  seed registry, P2b pilot checksum, CR1 result checksum, protocol, and
  artifact contract; and
- pass a non-scientific I/O/checkpoint/replay smoke that reports no effect
  sizes, arm ordering, event rates, or candidate differences.

The estimated cost is 11--13 CPU-hours and approximately 1.8--2.0 GB of disk,
based on the sealed full CR1 campaign. Launch requires at least 2.5 GB free and
a declared CPU budget of at least 15 hours. Once launched, the phase completes,
replays, seals, and reports; it is not killed mid-phase because an estimate was
imperfect.

After the checksum-sealed result, stop for review. Do not automatically launch
CR4, CR5, CR6, CR7, a new rule search, or any rescue analysis.

## Claim boundary

A pass may support that the simple outgoing catalytic-support rule
prospectively controls the operational simulated `JOINT_BREAK_RUN3` process in
both Codex candidates and that a physically interpretable local rule captures
a measurable fraction of the frozen predictor's molecular control effect.

It cannot establish strict-eight control, life, agency, biological memory,
error correction, autonomous organization, an autonomous attractor, real
prebiotic chemistry, Phi/PhiID intervention, or a universal origin-of-life
mechanism.
