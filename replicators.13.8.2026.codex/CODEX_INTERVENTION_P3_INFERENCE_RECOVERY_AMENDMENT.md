# Codex P3 inference-routing recovery amendment

Status: to be checksum-sealed after the P3 run stopped and before any P3
checkpoint outcome, effect size, arm rate, interval, p-value, candidate
comparison, or scientific verdict is loaded or calculated.

## Recorded failure

P3 completed all 51,200 registered primary futures and all 51,200 replay
futures across 400 restored states.  It then stopped at the first inference
call with:

```text
ValueError: missing registered arms: ['RANDOM']
```

No result bundle was written or sealed.  The completed primary and replay
checkpoint trees remain under their original campaign contracts.

## Static diagnosis

The P3 scientific contract registers these arms:

```text
LOOSEN, TIGHTEN, RANDOM_SURGERY, NOOP
```

The shared inference routine accepts the semantic random-control arm as a
keyword, but defaults that keyword to `RANDOM`, the name used by the molecular
intervention phases.  The P3 caller did not override the default.  Validation
therefore stopped before indexing arms or calculating any scientific estimate.

## Only repair

Both the primary inference and its exact written-artifact readback must call
the already frozen inference routine with:

```text
random_arm = "RANDOM_SURGERY"
```

The registered `LOOSEN - TIGHTEN` contrast, random-versus-no-op specificity
comparison, whole-matrix bootstrap, whole-matrix sign randomization, Holm
correction, equivalence margin, effect-ratio rule, and every other analysis
remain unchanged.

## Recovery contract

This is a source-additive, zero-future recovery.  It must:

- preserve the original registration and the prospective P3 lifecycle
  amendment unchanged;
- seal this amendment and aggregate hashes of both completed checkpoint trees
  before unpickling any scientific checkpoint;
- reconstruct the deterministic natural cohort;
- load exactly 400 primary and 400 replay state checkpoints;
- generate and regenerate zero intervention futures;
- require exact replay;
- require checkpoint aggregate hashes to remain unchanged;
- round-trip the complete inference with `RANDOM_SURGERY` explicitly routed;
- checksum-seal the result and a linked lifecycle audit; and
- stop before any confirmation campaign.

No simulator, beta surgery, matrix, state, seed, endpoint, arm, future,
prediction, bootstrap draw, randomization draw, multiplicity correction,
margin, gate, report boundary, or phase-advancement rule changes.

The resulting P3 evidence remains a 40-matrix developmental pilot.  Positive,
mixed, or negative results must be reported unchanged and cannot be called an
untouched confirmation.
