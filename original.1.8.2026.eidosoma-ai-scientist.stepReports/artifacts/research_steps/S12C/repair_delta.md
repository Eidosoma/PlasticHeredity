# S12C preregistered repair delta

## Top summary

- **Research step ID:** S12C (`E01-S12C-SOURCE-EQUIVALENCE-CONFIRMATION-v1.0.0`)
- **Completion status:** PREOUTCOME_DESIGN_FROZEN; development, confirmation, and GARD outcomes not yet opened.
- **Artifacts written:** `preregistration.yaml`, `preregistration_record.json`, `immutable_input_audit.json`, `source_snapshot_manifest.json`, `safe_lattice_reference.json`, and this repair delta.
- **Validation result:** PASS if and only if the accompanying preregistration record has `success: true` and the design commit is clean and pushed.
- **Outcome classification:** Pending; this document contains no confirmation or GARD scientific result.
- **Caveats or blockers:** The repair is informed by the known S12B singular failure. Confirmation therefore uses a separate untouched root and remains inaccessible until implementation lock.
- **Recommended next action:** Run only the frozen development suite, audit the wrapper-only delta, lock and push the implementation, and then run untouched confirmation. Do not open GARD input before unanimous confirmation.

## Exactly one permitted repair

S12B used a vectorized cross-correlation block for IIGR lagged MI. On the preserved exact-duplicate fixture it differed from the pinned nested `scipy.stats.pearsonr` loop by about `3.64e-17`; the resulting degenerate Fiedler split changed, and the wrapper reached a singular reduced covariance while the pinned source did not. S12C replaces only that vectorized IIGR MI calculation with the pinned source's pairwise loop, assignment order, and significance comparison. It does not regularize IIGR, change exception policy, alter the Fiedler algorithm, change PhiRL, or weaken any gate.

## Immutable boundaries

S12B remains failed and byte-exact. Both public commits, their file hashes, safe-lattice JSON, S12 trajectories/labels/preprocessing, modes, statistics, tolerances, and classifications remain frozen. A confirmation exception cannot be relabeled as equivalent to source eligibility. Any confirmation failure permanently closes this repair path without GARD access or another repair.
