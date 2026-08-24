# S19-L10 technical repair 001

## Concise top summary

- **Research step ID:** `S19-L10`.
- **Completion status:** additive technical repair locked before rerun; repaired outcome pending.
- **Artifacts written:** preserved initial regeneration validation/table/trajectory/runtime evidence, repair decision JSON, and this repair note.
- **Validation result:** the initial 400/400 trajectory replay passed; 13/14 table hashes passed; the sole failed table had the same columns and zero different cells after diagnostic alignment but a different column order.
- **Outcome classification:** pending complete repaired regeneration; no scientific classification is changed by this lock.
- **Caveats or blockers:** the repair is post-outcome but explicitly human-authorized; it may canonicalize schema order only and may not change any scientific code, method, value, seed, or gate.
- **Recommended next action:** commit and push this repair, verify the clean release gate, rerun all 400 trajectories and all tables in a fresh cache, then accept results only if every exact value/schema gate passes.

The initial failure is preserved under `*_failed_attempt_001.*`. Repair 001 fixes only the omission of column-order canonicalization in the exact table comparator. Lexicographic column order is deterministic and independent of scientific values. The original scientific runner/core/config at commit `e257cad4263ee63d733c37f041ee6994eeb7e385` remain unchanged.
