# Chapter 5 Phi-r procedural amendment 001

Date: 2026-08-18

## Scope

This amendment repairs only the integrity serialization used between a
scientific worker process and its parent checkpoint writer. It changes no
scientific question, simulator contract, matrix or future seed, cohort size,
arm, intervention selector, endpoint, Phi-r instrument, clock, window,
threshold, inference rule, confirmation barrier, or claim boundary.

The original sealed registration was
`fedbe1184b1b202411e511725efb0a086f305c5d5575614c61d4cc790b803899`.
Its first detached pilot attempt stopped before writing any generated or replay
matrix checkpoint and before analysis. No arm effect, event rate, ordering,
candidate difference, or other scientific result was available when this
repair was designed.

## Failure record

- Failed-attempt status SHA-256:
  `602139ef77ef6bfa4409d6f287dab659e50a8e9604873e70638885c289bbfa8d`.
- Failed-attempt log SHA-256:
  `870693cdbdb853310ef7d256996054e60030cdc316f69818822a98154f881f2e`.
- Original registration manifest SHA-256:
  `156f045de133e7893c2a4a1a885eef26c8a0b83a5e96da7be7d4dba5665ed777`.
- Generated checkpoints: 0 of 24.
- Replay checkpoints: 0 of 24.
- Failure: the parent rejected the first returned worker object because a
  digest computed from raw pickle bytes differed after multiprocessing
  serialization.

## Repair

The batch integrity digest is now calculated from a canonical, key-sorted JSON
encoding of the batch's scientific values with the digest field blanked.
Consequently it is invariant to pickle memo/reference layout while still
changing when scientific content changes. A regression test explicitly sends
an alias-rich batch through a pickle round trip and requires exact digest
stability. Worker validation now reports matrix-ID and content-digest failures
separately.

The original failed registration, validation, smoke, work tree, and log are
retained in versioned archive paths. Validation, registration, and the
non-scientific smoke are regenerated under the repaired source seal before the
same 24-matrix pilot is restarted. Reusing the sealed pilot seeds is permitted
because no scientific checkpoint or result was exposed.
