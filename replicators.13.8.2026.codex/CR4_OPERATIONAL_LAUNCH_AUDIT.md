# CR4 operational launch audit

Registration: `c088f787e11fcd1474adb886e1978bd79ed5c5a497e38815833ddf4cc6f9c9f5`

This is an operational provenance note.  It does not amend the sealed CR4
scientific protocol, code, model, seed registry, cohort, arms, endpoint,
inference, or gates.

## What happened

1. The registered CR4 command was launched through `setsid` with 14 workers.
2. A sandbox-scoped process query could not see the host process and the log
   still showed cohort construction.  The launch was therefore mistakenly
   judged to have ended.
3. At that instant the work prefix contained zero `state_*.pkl` scientific
   checkpoints.  The empty prefix and zero-length timing/log wrappers were
   removed and the identical registered command was relaunched using
   `setsid --fork`.
4. A later host-scoped audit showed that the original process had survived.
   The newer duplicate process group was terminated immediately, before any
   scientific inference or result inspection.  The original process was
   retained.

## Scientific consequences and safeguards

- Both processes used the same immutable registration, source hashes, frozen
  model, matrix/state seeds, arm-selection seeds, and arm-free future seeds.
- During the brief overlap they could only generate byte-deterministic copies
  of the same state checkpoints.  Checkpoint writes are atomic and keyed by
  state index.
- No matrix, state, branch, adverse outcome, or extinction was inspected,
  selected, replaced, or excluded.
- No endpoint rates, arm ordering, effect sizes, candidate differences,
  bootstrap results, or randomization results were opened.
- The retained campaign must still complete its registered full replay.  Its
  state, surgery, endpoint, and process digests must agree exactly; otherwise
  CR4 fails its integrity gate.
- The duplicate consumed extra CPU but cannot change a scientific value.

This launch incident must be reported with the final CR4 result.  It is not a
scientific protocol deviation, but it is retained as an operational deviation
for complete provenance.

## Completion audit

- Final campaign state: `sealed_complete`.
- Primary futures: 640,000 of 640,000.
- Exact replay futures: 640,000 of 640,000.
- Exact replay, surgery audit, and written-artifact readback: all passed.
- Every file in the final result bundle passed its sealed `SHA256SUMS` check.
- The original process's log and `/usr/bin/time` file handles referred to files
  unlinked during the mistaken zero-checkpoint relaunch.  Consequently the
  visible log is incomplete and the timing file is empty.  This loses an
  operational CPU-timing measurement, not a scientific state, future, result,
  or integrity record.  Durable checkpoint counters, the result manifest,
  replay audit, readback audit, and checksum seal are complete.
