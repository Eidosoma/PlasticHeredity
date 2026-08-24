# S19-L10 mandatory human-review decision summary

## Concise top summary

- **Research step ID:** `S19-L10`.
- **Completion status:** COMPLETE; scientific work stopped at the mandatory human-review boundary.
- **Artifacts written:** the complete L10 evidence package, 12 figures, full report, machine-readable classification, validation records, and hashes.
- **Validation result:** PASS_ALL_FIXTURES_SEED_FIREWALL_400_TRAJECTORY_REPLAYS_14_RESULT_TABLE_REPLAYS_IMMUTABILITY_SOURCE_SCOPE_RUNTIME_STORAGE_AND_HASH_GATES.
- **Outcome classification:** `RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED`; `EXPLORATORY_NON_SUPPORT`, `AUTHOR_AMBIGUITY_UNRESOLVED`, `NOT_PROMOTABLE`.
- **Caveats or blockers:** exploratory completed-run labels; no author-code identity, prospective prediction, or causal-control inference.
- **Recommended next action:** human review only; no later step is active.

## Decisive results

- R1 / CANDIDATE_2: occupancy=0.398005726163988, persistence=266.4691358024691, consistency=0.8608667105836663, onset=106.08641975308642.
- R1 / CANDIDATE_3: occupancy=0.40390449649255405, persistence=282.2289156626506, consistency=0.8807014670862442, onset=95.08433734939759.
- R2 / CANDIDATE_2: occupancy=0.27160822999462303, persistence=169.07070707070707, consistency=0.7698344543978757, onset=127.15277777777777.
- R2 / CANDIDATE_3: occupancy=0.28192906941310597, persistence=185.27272727272728, consistency=0.7894534544283353, onset=132.92753623188406.

Promoted lead IDs: `[]`. The full promotion gate—not occupancy alone—was applied. L09 remains failed closed and unchanged. S18's prospective and causal conclusions remain unchanged.

## Technical-repair disclosure

The first exact-regeneration attempt passed 400/400 trajectories and 13/14 tables; the sole failed table had zero different cells after diagnostic column alignment but a different scheduling-dependent column order. After explicit human authorization, `S19-L10-TECHNICAL-REPAIR-001` preserved that failure, changed only column-order canonicalization, reran the complete scope in fresh caches, and passed 400/400 trajectories and 14/14 tables with no scientific value change. The repaired validation permits the locked scientific classification; it does not make the negative result more favorable.
