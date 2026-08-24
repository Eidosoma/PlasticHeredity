# CR6 administrative readback amendment

Status: specified after the original CR6 process stopped and before it was
resumed. This amendment does not alter a scientific definition, simulation,
seed, cohort, intervention, model, outcome, contrast, confidence interval, or
gate.

## Trigger

The originally sealed CR6 runner completed all 160 primary state batches and
all 160 exact-replay state batches for `POS_A_M4_S5`. It then stopped before
creating a regime artifact. The failure occurred in the written-artifact
readback audit. No later regime had started and no scientific effect size,
arm mean, candidate difference, confidence interval, or gate result had been
written or inspected.

The primary and replay checkpoint directories each contain exactly 160 state
files and report 30,720 of 30,720 futures complete. The final CR6 result path
does not exist.

## Root cause

The writer adds the regime identifier to each in-memory matrix-effect row
before writing it:

```python
for row in matrix_rows:
    row["regime"] = regime
```

The readback function recomputes the same matrix rows but compares them before
adding that administrative identifier. The numerical fields are therefore
unchanged, but dictionary equality must fail because one side has an extra
`regime` key.

## Fixed operation

The amendment supplies a replacement readback callback that performs the
original computation and then applies the identical deterministic label to
the recomputed rows before equality is tested:

```python
for row in observed_rows:
    row["regime"] = regime
```

No numerical value is rounded, changed, discarded, or accepted with a weaker
tolerance. The audit remains exact equality after JSON normalization.

The original CR6 source file and registration remain byte-for-byte unchanged.
The amendment is an explicit process-local overlay used only by the artifact
readback call. The original runner continues to construct cohorts, select
edits, simulate futures, replay futures, compute inference, apply gates, write
artifacts, and enforce the mandatory CR7 stop.

## Validation and resumption

Before resumption, the amendment must:

1. verify the original CR6 registration, validation, and smoke checksums;
2. verify the stopped-run log and checkpoint contracts;
3. confirm 160 primary and 160 replay checkpoints for the first regime;
4. confirm that no final result or first-regime staging artifact exists;
5. reproduce the original false readback failure on an artificial fixture;
6. show that adding only the missing deterministic label makes the same
   fixture pass exact numerical and row readback;
7. pass the full repository test suite;
8. seal amendment source hashes and stopped-run provenance.

The resumed run reuses the original registration ID, model, seed domains,
campaign contract, and checkpoints. It may not retry, replace, or select a
scientific matrix based on outcomes. It must still replay every future and
pass every original CR6 gate and checksum. The final top-level result records
both the original registration ID and this amendment ID.

## Claim boundary

This correction can restore the intended exact artifact audit. It cannot turn
a failed scientific gate into a pass, change an effect estimate, expand the
registered claim, or authorize CR7 automatically.
