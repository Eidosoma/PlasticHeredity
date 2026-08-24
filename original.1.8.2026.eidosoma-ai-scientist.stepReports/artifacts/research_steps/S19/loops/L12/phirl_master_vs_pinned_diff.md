# PhiRL current master versus pinned commit

## Concise top summary

- **Research step ID:** `S19-L12`.
- **Completion status:** SOURCE COMPARISON COMPLETE.
- **Artifacts written:** `source_snapshot_manifest.json`, `phirl_repository_tree.json`, `phirl_commit_history.csv`, and `phirl_function_blame.csv`.
- **Validation result:** PASS — remote `master`, local checkout, and the pinned commit are all `a6d1d0d18c7551302724b7158c6ccdc4d3a33373` with tree `e59fa8e311c2f727724acf3c1f1885dc8d840ee5`.
- **Outcome classification:** `DIRECT_PUBLIC_CODE`; no version drift exists to explain E01 discrepancies.
- **Caveats or blockers:** Equality to public master does not identify the unavailable GARD-paper implementation. No tag or alternate public branch supplies GARD-specific code.
- **Recommended next action:** Audit the internal commit lineage and paper/source behavior; do not infer author-code identity.

The diff is empty because the current remote master and pinned commit are identical. The complete public tree contains 9 files. PhiRL has 35 commits across all local refs; no tag is present. The meaningful version boundary is internal: slow bidirectional lagged MI and `local_phi_r` existed first, public `emergence = synergy + causation` was exposed later, and the fast-MI plus trace-scaled covariance regularization path arrived in November 2025.
