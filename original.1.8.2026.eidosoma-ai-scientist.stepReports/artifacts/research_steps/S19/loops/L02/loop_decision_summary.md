# S19-L02 — Replicator-definition temporal-fingerprint reconstruction

## Concise top summary

- **Research step ID:** S19-L02
- **Completion status:** COMPLETE; mandatory human-review boundary reached; S20 and every later loop remain inactive
- **Artifacts written:** 40 compact L02 evidence/report files plus append-only S19 root-ledger updates
- **Validation result:** PASS_ALL_LOCK_REPLAY_IMMUTABILITY_STORAGE_AND_REGENERATION_CHECKS
- **Outcome classification:** EXPLORATORY_CONSTRAINING_NO_PROMOTABLE_LEAD
- **Caveats or blockers:** No label passed the complete source-grounding and joint-fingerprint promotion gate; the two-loop default is S20 closeout-only. The paper does not uniquely specify clustering, recurrence, threshold, reference, or molecular/generation alignment; completed-run clustering is future-dependent.
- **Recommended next action:** Human review must choose the next program action. Current default: `ACTIVATE_S20_CLOSEOUT_ONLY_TWO_CONSECUTIVE_LOOPS_WITH_NO_PROMOTABLE_LEAD`. Do not begin it automatically.
- **Lay summary:** This loop tested whether the gap between roughly 98% replication in the current adjacent-similarity label and the paper's roughly 88% state could be explained by a genuinely different definition of a replicator. It compared four fixed definitions and judged the whole temporal pattern—not occupancy alone. The analysis did not tune a threshold, generate simulations, or use causal-emergence results to pick a label.


## Decision evidence

Promoted leads: **none**.

| Cand.   | Label             | Onset mode   |   Distance gain |   Closer dims | Structure improved   |   Bootstrap low |   Bootstrap high | All LOO improved   |
|:--------|:------------------|:-------------|----------------:|--------------:|:---------------------|----------------:|-----------------:|:-------------------|
| C02     | Dominant centroid | RAW          |         -2.4575 |             0 | False                |          7.0241 |           8.4902 | False              |
| C02     | Dominant centroid | NORMALIZED   |         -2.4171 |             1 | True                 |          6.9127 |           8.3496 | False              |
| C02     | Euclidean cluster | RAW          |         -0.9527 |             1 | True                 |          2.6167 |           3.4564 | False              |
| C02     | Euclidean cluster | NORMALIZED   |         -0.9596 |             1 | True                 |          2.6462 |           3.4827 | False              |
| C02     | Historical T1     | RAW          |         -2.3416 |             0 | False                |          6.7261 |           8.0761 | False              |
| C02     | Historical T1     | NORMALIZED   |         -2.33   |             1 | True                 |          6.6998 |           8.0282 | False              |
| C03     | Dominant centroid | RAW          |         -2.648  |             0 | False                |          7.4138 |           8.8449 | False              |
| C03     | Dominant centroid | NORMALIZED   |         -2.6023 |             1 | True                 |          7.2956 |           8.6341 | False              |
| C03     | Euclidean cluster | RAW          |         -0.9913 |             2 | True                 |          2.6293 |           3.4971 | False              |
| C03     | Euclidean cluster | NORMALIZED   |         -0.9978 |             2 | True                 |          2.6672 |           3.5226 | False              |
| C03     | Historical T1     | RAW          |         -2.5209 |             0 | False                |          7.0813 |           8.424  | False              |
| C03     | Historical T1     | NORMALIZED   |         -2.4741 |             1 | True                 |          6.9835 |           8.2354 | False              |

## Human choice required

The loop is frozen. The human reviewer must issue a new decision before any further scientific work. The current scientific recommendation is `ACTIVATE_S20_CLOSEOUT_ONLY_TWO_CONSECUTIVE_LOOPS_WITH_NO_PROMOTABLE_LEAD`. S20 remains defined but inactive.
