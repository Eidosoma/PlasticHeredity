# IIGR–PhiRL source-lineage map

## Concise top summary

- **Research step ID:** `S19-L12`.
- **Completion status:** SOURCE LINEAGE AUDIT COMPLETE.
- **Artifacts written:** source hashes, complete PhiRL history, function blame, and safe-lattice equivalence.
- **Validation result:** PASS — IIGR `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7` and PhiRL `a6d1d0d18c7551302724b7158c6ccdc4d3a33373` share the byte-identical 16-node lattice and closely corresponding local-Phi functions; their repositories have no Git-parent relationship.
- **Outcome classification:** `SOURCE_LINEAGE_INFERENCE`, not author implementation identity.
- **Caveats or blockers:** IIGR is a GRN application and PhiRL is an RL-representation application. Neither tree contains the GARD prediction or intervention pipeline.
- **Recommended next action:** Keep inherited operations and later PhiRL changes distinct in the executable data-flow audit.

IIGR predates PhiRL and supplies the closest public ancestry for the lattice, local Gaussian entropy, local ΦID inversion, `local_phi_r`, lagged mutual-information graph, Fiedler partition, and partition averaging. IIGR's terminal commit is explicitly named “fix phir bug”; S12B/S12C verified the corrected atom set. PhiRL initially copied the slow lineage, later exposed both `integrated` and `emergence`, and then introduced the fast-MI and covariance-regularized path. The two lattice pickles are byte-identical. Structural correspondence and chronology justify a lineage inference, but public Git metadata does not prove a direct code-copy event or the pipeline used for the GARD paper.
