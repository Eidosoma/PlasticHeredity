# P1 readback-recovery amendment

Status: preregistered after the P1 process stopped, but before any P1 effect size, arm rate, candidate comparison, p-value, confidence interval, or pilot verdict was read or reported.

## Preserved execution

The original registration is `f61e0340dcd8c9ae6b606c8133ca3d8fb1de2e13fe863719aa67b649e8b74531`. Under that registration, all 400 P1 states completed their 51,200 primary F12 futures and all 400 states completed the 51,200 exact-replay futures. Both checkpoint sets are retained. The atomic result writer removed its temporary output after the exception, so no result bundle or scientific report was sealed. P2 was not launched.

The registered simulator, endpoint, molecular edits, exhaustive edit scores, selected arms, seed keys, matrices, restored states, future branches, replay branches, bootstrap draws, randomization signs, multiplicity correction, equivalence margins, gates, and claim boundaries remain unchanged. The original scientific source files and original registration remain unchanged.

## Static diagnosis

The failure occurred in the non-scientific round-trip audit after inference and artifact writing had begun. The generation path adds the derived Boolean field `pilot_eligibility` to the primary metrics. The readback path recomputes all inference fields but compared its dictionary before adding that one derived field. Therefore, the dictionaries necessarily differ even when every underlying outcome and statistical value is identical.

The failed log ends with:

```text
ValueError: round-trip intervention inference changed
```

No scientific output was inspected to make this diagnosis; it follows directly from the sealed source and traceback.

## Frozen repair

The repair is source-additive. A new recovery module will:

1. verify the original registration and all original scientific source hashes;
2. verify aggregate hashes of every completed generation and replay checkpoint;
3. reconstruct the same deterministic natural cohort;
4. load all completed checkpoints without invoking a missing-state worker;
5. recompute the same registered inference with the same registered draws;
6. derive readback pilot eligibility as

   ```text
   readback pilot eligibility =
       readback pilot eligibility without replay
       AND exact replay
   ```

7. compare the complete readback dictionary and matrix-effect table;
8. verify that checkpoint hashes are unchanged;
9. seal the full result and reports; and
10. stop without launching P2.

No intervention future may be generated or regenerated during recovery. If a state checkpoint is missing, invalid, or changed, recovery must fail rather than fill it. The recovery source and this amendment must be checksum-sealed before any checkpoint outcome is loaded by the recovery command.

## Interpretation

This amendment can repair only the artifact/readback plumbing. It cannot change or rescue the P1 scientific verdict. A negative, mixed, or positive result must be reported exactly as recovered. Pilot eligibility remains developmental and is not an untouched cross-clean-room confirmation.

