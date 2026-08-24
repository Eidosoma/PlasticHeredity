# PhiRL components missing for the GARD paper pipeline

## Concise top summary

- **Research step ID:** `S19-L12`.
- **Completion status:** COMPLETE SOURCE-GAP AUDIT.
- **Artifacts written:** executable data-flow, function, numerical, atom, leakage, and source-lineage registries.
- **Validation result:** PASS — every registered PhiRL function was located and traced at the pinned/current commit.
- **Outcome classification:** `PUBLIC_CODE_MISSING` for the GARD-specific end-to-end pipeline.
- **Caveats or blockers:** Absence from the inspected public history is not proof that no private implementation exists.
- **Recommended next action:** Use the public source only for pinned component semantics; author code is required to identify the paper's complete pipeline.

No public PhiRL branch, tag, deleted path recovered through Git history, or current file implements GARD simulation, GARD preprocessing from count trajectories, the paper's self-replicator label, Figure 2 unequal-length aggregation, Figures 3–4 GARD statistics, the Figure 5 sequence tensor/MLP, alternative input proportions, spike-descriptor analysis, hypothetical post-fission action scoring, max/control/min GARD intervention trajectories, or Table 1 outcome aggregation. IIGR and BreakingGRNMemories provide related information-theory ancestry and application patterns, not these missing GARD components.

Public PhiRL does establish a specific component chain: active-variable filtering, z-scoring, fast lagged-MI construction, a noise-connected unnormalized Fiedler split, arithmetic partition averaging, complete-trajectory Gaussian fitting, local PhiID, and two distinct exported scalars called `integrated` and `emergence`. That chain cannot by itself decide which scalar, label, tensor, intervention scorer, or denominator generated the manuscript figures.
