# Codex P3 lifecycle amendment

Status: this amendment must be checksum-sealed before the first P3 scientific
matrix is generated.

## Scope

P3 is the already registered CR4 fixed-composition beta-surgery pilot.  Its
scientific contract remains exactly the contract sealed under original
registration:

```text
f61e0340dcd8c9ae6b606c8133ca3d8fb1de2e13fe863719aa67b649e8b74531
```

The amendment changes only the result-readback lifecycle.  The original runner
adds the replay-dependent `pilot_eligibility` field to generation metrics but
omits that derived field when recomputing metrics from written arrays.  P1 and
P2 exposed this defect only after all primary and replay futures had completed.

For P3, readback must reconstruct the missing field before exact dictionary
comparison:

```text
readback pilot eligibility =
    readback pilot eligibility without replay
    AND written exact-replay status
```

No simulator, beta surgery, matrix, state, seed, branch, endpoint, arm, model,
bootstrap, randomization, margin, gate, report boundary, or stop rule changes.

## Unchanged P3 scientific design

- 40 fresh matrices shared across candidates 02 and 03.
- Natural landmarks 20, 35, 50, 65, and 80.
- 400 restored states.
- 32 F12 futures per arm and state, with halves A=0--15 and B=16--31.
- Arms: `LOOSEN`, `TIGHTEN`, `RANDOM_SURGERY`, and `NOOP`.
- Composition and observed history identical across arms at launch.
- `TIGHTEN` and `LOOSEN` alter every present-present beta edge by a common
  multiplicative factor at exact Frobenius norm
  `0.05 * ||beta[P,P]||_F`.
- `RANDOM_SURGERY` changes the same number of independently selected edges,
  preserves positivity, uses a balanced log perturbation, and matches the exact
  norm.
- New P3 seed domains already frozen by the original registration.
- 4,096 whole-matrix bootstrap draws and 4,096 paired sign randomizations.
- Holm correction across the four candidate-by-half cells.
- Complete deterministic replay and mandatory stop after sealing.

The externally clarified Fable C3 outgoing-rule orientation does not change P3:
the complete present-present edge set is the same under matrix transposition.

## Provenance and reporting

The original checksum-sealed source tree is not edited.  This additive module
temporarily replaces only the in-memory readback callback, runs the original P3
entry point, restores the callback, and creates a separate checksum-sealed audit
linking the P3 result to this amendment.

P3 remains a developmental pilot.  It cannot launch the 160-matrix mechanism
confirmation automatically.  The result must be reviewed together with P1 and
corrected P2b before one mechanism is separately frozen for confirmation.
