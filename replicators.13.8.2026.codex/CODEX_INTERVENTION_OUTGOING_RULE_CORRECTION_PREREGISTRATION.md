# Codex P2b outgoing-rule correction preregistration

Status: implementation and validation must be checksum-sealed before any P2b
scientific matrix is generated.

## Why this is a correction experiment

The sealed Codex P2 pilot implemented the earlier prose directive as incoming
target support under Codex storage `beta[target, catalyst]`:

```text
incoming = beta @ x
```

After P2 was sealed, the independent Fable implementation supplied the frozen
operational definition used consistently by its C3, G4, and G5 analyses:

```text
x = n / N
outgoing = x @ beta
```

Fable and Codex both store rows as targets and columns as catalysts.  Therefore
the Fable rule in Codex column-vector notation is exactly:

```text
outgoing = beta.T @ x
```

It measures how strongly candidate catalyst type `t` supports the molecular
types currently present.  It is not `beta @ x`, which measures how strongly
target type `t` is supported by the catalysts currently present.

The sealed P2 result remains valid for its registered incoming-support rule and
is retained at full prominence.  It is reclassified as an incoming-support
negative control, not as a test of Fable C3.  No P2 state, seed, edit, branch,
outcome, interval, p-value, report, or checksum is changed.

## Frozen P2b question and rule

P2b asks whether the externally disambiguated outgoing catalytic-influence rule
causally controls Codex `JOINT_BREAK_RUN3`.

For composition `n`, mass `N`, and Codex beta storage
`beta[target, catalyst]`, define:

```text
x = n / N
outgoing[t] = sum_i x[i] * beta[i, t]
            = (beta.T @ x)[t]
```

Because every legal edit preserves `N`, using `n` or `x` gives identical
rankings.  The implementation nevertheless uses normalized `x` to reproduce
the frozen rule literally.

The four arms are:

- `RULE_UP`: remove the most influential present type and add the least
  influential different type.
- `RULE_DOWN`: remove the least influential present type and add the most
  influential different type.
- `RANDOM`: one uniformly sampled legal substitution from an independently
  domain-separated selection stream.
- `NOOP`: unchanged composition.

Every selection is made by exhaustive enumeration of legal one-molecule
substitutions.  Ties use the existing deterministic lexicographic ordering.
The frozen predictor is scored descriptively but neither selects the rule nor
changes any gate.

## Fresh pilot cohort

- 40 entirely new catalytic matrices shared across candidates.
- Existing sealed candidate 02 and 03 simulator contracts without alteration.
- Natural landmarks 20, 35, 50, 65, and 80.
- 400 restored states.
- 32 F12 futures per arm and state.
- Fixed halves A=0--15 and B=16--31.
- 51,200 primary futures and a complete 51,200-future deterministic replay.
- No intervention-future retries and no matrix replacement.
- New cohort, selection, future, bootstrap, and randomization seed domains that
  are distinct from every original P0/P1/P2/P3 domain.

For a candidate, matrix, landmark, and branch index, arm identity is absent
from the future seed.  Arms receive common random streams, not guaranteed
identical realized paths after divergence.

## Frozen analysis and gates

The catalytic matrix is the inference unit.  Candidates and branch halves are
reported separately.  All states, landmarks, arms, and halves belonging to one
matrix stay together in every draw.

Use 4,096 shared whole-matrix bootstrap draws, 4,096 shared paired sign
randomizations, and Holm correction over the four candidate-by-half cells.

The full registered cell gate is unchanged from the original intervention
program:

1. `RULE_UP - RULE_DOWN > 0`.
2. Its 95% matrix-bootstrap lower bound is positive.
3. Holm-adjusted matrix-randomization `p < 0.05`.
4. `RULE_UP > NOOP` with a positive bootstrap lower bound.
5. `NOOP > RULE_DOWN` with a positive bootstrap lower bound.
6. `RANDOM` is equivalent to `NOOP` by the `+/-0.025` TOST margin and its
   absolute point difference is no greater than 25% of the targeted contrast.
7. Exact replay passes.

The developmental pilot-eligibility rule is also unchanged: positive
`RULE_UP - RULE_DOWN` and absolute `RANDOM - NOOP <= 0.025` in all four cells,
plus exact replay.

The already observed, prediction-only outgoing-rule diagnostic is disclosed as
post-P2 motivation and is not an outcome or a gate.  No P2b future has been
observed when this document and its implementation are sealed.

## Lifecycle correction

The versioned P2b runner reconstructs the replay-dependent
`pilot_eligibility` field before readback dictionary comparison.  This is the
only lifecycle repair.  The original checksum-sealed runner and its recovered
P1/P2 artifacts remain unchanged.

The two checksum-sealed historical tests asserting that atomic P1/P2 output
directories did not yet exist are retained verbatim.  An additive current-state
pytest hook marks exactly those superseded pre-recovery assertions as expected
failures; all remaining tests must pass.

## Stop and claim boundary

P2b stops after complete replay, reporting, and checksum sealing.  It does not
launch P3 or an untouched confirmation.

A successful P2b pilot would support only developmental eligibility of the
correct outgoing-rule family for a later untouched confirmation.  It cannot by
itself establish cross-clean-room replication.  It does not support Phi/PhiID,
strict-eight control, biological memory, autonomy, life, real chemistry, or a
universal origin-of-life mechanism.
