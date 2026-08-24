# S12B pinned-source audit

## Top summary

- **Research step ID:** S12B (`E01-S12B-PIGOZZI-SOURCE-CODE-AUDIT-v1.0.0`)
- **Completion status:** Pre-outcome source audit complete; scientific execution not yet started.
- **Artifacts written:** `preregistration.yaml`, `preregistration_record.json`, `immutable_input_audit.json`, `source_snapshot_manifest.json`, `source_audit.md`, and `safe_phi_lattice.json`.
- **Validation result:** PASS — both commits, trees, three required files per repository, raw-pickle identity, safe conversion, regularization ancestry, and prior S10–S12 immutability matched the frozen design.
- **Outcome classification:** Pending; no S12B GARD outcome was inspected for this audit.
- **Caveats or blockers:** This is `SOURCE_INFORMED_RECONSTRUCTION`, no license file was found, the original GARD author implementation remains unavailable, and the raw pickle is barred from scientific execution.
- **Recommended next action:** Commit and push the complete pre-outcome design, then run source-equivalence validation; stop before GARD processing if any equivalence gate fails.

## Pinned identities and source behavior

- IIGR: commit `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`, tree `b0baf451876f4c8760f25096b7d426add68f6881`. `main.py:26–30` defines z-score → global-signal regression → lag-one residualization. `main.py:108–122` defines alpha=1/no-Bonferroni lagged MI, the unnormalized Fiedler split, partition averaging, corrected `local_phi_r`, and diagnostic `emergence`. `information.py:27–32`, `43–53`, `56–118`, `121–148`, and `151–201` provide the traced implementations.
- PhiRL: commit `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`, tree `e59fa8e311c2f727724acf3c1f1885dc8d840ee5`. `main.py:28–53` removes dimensions at or below `1e-8`, z-scores, applies fast lagged MI, partitions, averages, and decomposes. `information.py:47–59` applies `epsilon=1e-6` trace-scaled covariance regularization; `information.py:189–244` supplies the fast MI and unnormalized Fiedler behavior. Regularization commit `9030b598f436cd23c39a3c3fc312ff79c79fb2ad` is an ancestor of the pinned commit.
- Both lattice pickles are byte-identical SHA-256 `66cd662640079e9a2a8bc172250b124d59945fd805b7f91d5588e2f7d1d7ea03`; the safe JSON artifact is SHA-256 `74ecca37f04201088d76a9e8ede7efe04bafebecff85a4882a44f03afbd23aa1`. The converter inspected every opcode and admitted only `dict`, `DiGraph`, and `NodeView` globals in a `python -I` disposable process.

## Relationship and license boundary

The public code concerns gene-regulatory and reinforcement-learning applications, not a released GARD simulator. It informs the local-Phi reconstruction but cannot establish the paper's unpublished GARD code, data layout, random-state ordering, or an author-primary method. Neither pinned tree contains a detected LICENSE or COPYING file, so no public-source payload is redistributed in S12B artifacts.
