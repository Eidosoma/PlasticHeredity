# Cluster Report: Attractors and Committors

## Cluster scope

`ATTRACTORS` covers S19-L18 through L37. It began with a pragmatic question—can anything observable before a frozen first-entry event predict that event?—and ended with a deeper diagnosis: the transition probability was real, but its original completed-run attractor destination was not a reproducible network-level object across independent lineages.

## Stage 1: broad precursor search

L18 defined an at-risk first-entry task with lower self-replicator occupancy than the permissive adjacent-H target. L19–L27 then tested a deliberately varied but bounded set of precursor representations:

- critical slowing, recurrence quantification, dynamic-mode summaries;
- topology, intrinsic dimension, and path geometry;
- survival and hazard formulations;
- random convolution features;
- event alignment;
- local GARD drift/diffusion operators;
- recurrence maps;
- transition-tube summaries.

Across development, validation, larger 400-matrix work, and both simulator candidates, none supplied a robust incremental early-warning coordinate beyond exact compositional similarity or ordinary path stability. A temporary event-aligned AUROC around 0.77/0.70 did not survive the required incremental gates. These null results narrowed the problem: either the representations discarded the relevant state information, or the single observed future was too noisy to define a learnable target.

## Stage 2: establish whether a committor exists

L28 restored 200 complete simulator states. Each state received 128 independent 32-observation futures, divided prospectively into two 64-branch halves. The target remained the matrix-specific basin reconstructed from the completed lineage and was therefore explicitly retrospective-basin-conditioned.

The empirical committor was highly reliable:

| Candidate | Corrected between-state variance | Split-half Spearman | Intermediate states (`0.1<q<0.9`) |
|---|---:|---:|---:|
| 02 | 0.0937 | 0.9260 | 42 |
| 03 | 0.0857 | 0.9325 | 36 |

Bootstrap lower bounds passed every preregistered gate. Existing observed-path, exact-H, operator, and recurrence-map summaries did not recover the signal.

![Independent branch-half empirical committors for restored simulator states.](figures/attractor_l28_split_half_reliability.png)

*Figure ATTRACTORS-1. L28 split-half reliability. Each point is a restored state; x and y are independent 64-branch estimates. Agreement near the identity line shows that the finite-horizon probability is state-dependent and reproducibly measurable, not merely noise from one realized future.*

This was a key conceptual advance: prediction failure did not imply absence of transition tendency. It showed that the chosen observed-prefix representations missed a measurable probability.

## Stage 3: a short propagator acts as a teacher

L29's exact one-step generator features were insufficient. L30 instead propagated each state for eight selected-clock observations under 64 independent branches and summarized the short ensemble. On held-out matrices, the eight-step moments ranked the H32 committor with Spearman 0.771 and 0.887 in candidates 02 and 03, far above exact-H and ordinary-path controls.

L31 then froze the L30 transformation and model and applied them without refitting to unused matrices. The result confirmed:

| Candidate | Untouched Spearman, eight-step moments | Exact-H control | Ordinary-path control |
|---|---:|---:|---:|
| 02 | 0.900 | 0.213 | 0.419 |
| 03 | 0.821 | 0.147 | 0.266 |

![Frozen short-propagator coordinate against independently estimated H32 committor on untouched matrices.](figures/attractor_l31_coordinate_confirmation.png)

*Figure ATTRACTORS-2. L31 untouched confirmation. The frozen L30 coordinate transfers to previously unused matrices and ranks independently re-estimated committors. The destination is still defined from the completed run, so this is a confirmed simulator shooting signal but not a prospective paper result.*

L32–L35 then tried to distill what the short teacher knew into static state, recent history, growth/fission phase, catalytic graph, and branch-mechanism summaries. These representations often fit development data but did not transfer consistently across candidates and confirmation cohorts. The result suggested either unmodeled stochastic information or a target that varied among lineages.

## Stage 4: test the destination itself

L36 generated two new independent reference lineages under each frozen catalytic matrix and changed only target provenance. Reference centroids were often similar in median, yet strict cross-lineage agreement ranged only about 0.60–0.75 and failed the registered transfer gates. The classification was `TARGET_BASIN_LINEAGE_SPECIFIC`.

L37 then allowed multiple recurring components. For every original/reference-A/reference-B triplet, two lineages formed a leave-one-lineage-out atlas and the third was scored against it, with unrelated-matrix and species-permutation controls. Although recurring components existed, they did not define a common basin family that reliably recognized and predicted entry in the held-out lineage.

![Decision matrix for multilineage attractor-family transfer.](figures/attractor_l37_multilineage_decision.png)

*Figure ATTRACTORS-3. L37 decision matrix. Green cells mark technical or local gates that passed; red cells mark the decisive failures of reciprocal rank, original-teacher transfer, multilineage attractor-family support, and an independent any-attractor committor. The result supports trajectory-specific targets within the tested scope.*

## Null results and contradictions

- Broad static and recent-history precursors did not generalize.
- Exact local generator moments were not enough; several nonlinear updates were required.
- A strong short-shooting teacher did not imply that its target was independently existing.
- Independent lineages under one catalytic matrix could share some recurrent structure without selecting one transferable centroid or atlas.
- Development fit was often much stronger than cross-matrix validation, exposing matrix-specific or target-specific relationships.

## Scientific interpretation

The empirical committor and short-propagator results are positive but conditional. They demonstrate that, given a basin defined from one completed lineage, the restored simulator state contains a reproducible probability of entry and that a few nonlinear updates reveal that probability. They do not demonstrate that the basin is a general self-replicator or attractor of the catalytic network.

The cross-lineage failure is therefore not a minor caveat. It changes the target question. A scientific event should be recognizable as it unfolds and should not require the evaluated lineage's completed future to name its destination. This insight motivated the next cluster's process-based definitions of recurrence, inheritance, disruption, and renewal.

## Evidence assessment

| Claim | Assessment |
|---|---|
| A reliable finite-horizon probability exists for the original target | Supported in both candidates |
| Existing static/observed-prefix summaries recover that probability | Not supported |
| Eight-step stochastic propagation recovers it | Confirmed on untouched matrices |
| The target is a stable same-matrix attractor | Not supported |
| Independent lineages share one transferable attractor family | Not supported |
| The result is a paper replication or first-replicator warning | No |

## Cluster conclusion

The attractor search produced an important paired result: a genuine, measurable state-dependent committor and a falsification of the destination used to define it. The scientifically productive response was not another centroid search. It was to replace privileged destinations with prospective dynamical processes.

