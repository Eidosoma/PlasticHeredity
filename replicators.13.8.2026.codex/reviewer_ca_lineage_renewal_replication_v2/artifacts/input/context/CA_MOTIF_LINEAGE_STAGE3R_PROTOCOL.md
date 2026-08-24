# Preregistered CA motif-lineage Stage 3R

Date frozen: 2026-08-23. This protocol is frozen after the completed negative
Stage-3 reference result and before any Stage-3R trajectory is generated. The
user authorized an ambitious repair round but selected a mandatory human review
before the final holdout.

## Question and evidence boundary

Stage 3R asks why the demonstrated 512-bin motif carrier loses ancestral-form
meaning when daughters rewrite it, and whether a fixed, lineage-blind repair
rule can close that parent-to-daughter semantic channel. The original form is
the primary endpoint. Target-free persistence of a causally renewable but
drifting lineage is a preregistered secondary endpoint and cannot turn a failed
strict-form result into strict Plastic Heredity.

No Wagner or Fable implementation source may be read, imported, hashed, or
executed. The fixed reader remains `motif_energy512-w32-s025-d32`: read strength
0.25 for sweeps 1--32, Rule 31649, process noise 0.002, a bitwise-identical
visible reset every generation, and checkpoints 1, 2, 4, 8, and 16. The carrier
is clipped to [-4,4]. Dead futures remain in all denominators.

## Cohorts and holdout lock

The 64 already exposed Stage-3 reference pairs are development data for
tomography and repair fitting only. The two exposed Stage-3 smoke pairs remain
the engineering cohort. Every pair exposed in Stages 1--3 is excluded from the
new scientific cohorts.

The remaining pairs are ordered under
`plastic-ca-motif-lineage-stage3r-v1`. The first 64 are candidate selection;
the next 96 are confirmation. Their identifiers and hashes are sealed before
selection. Confirmation cannot run in the same invocation: it requires a
separate `confirm` phase plus explicit authorization after human review. Thus a
failed screen leaves the confirmation trajectories untouched. At least 382
pairs remain unused after both cohorts.

Reference diagnosis uses 64 futures per history and 16 generations. Selection
uses 64 pairs, 32 futures per history, and 16 generations. Confirmation uses 96
pairs, 64 futures per history, and 16 generations. Smoke uses the two exposed
engineering pairs, two futures, and four generations.

## Semantic-drift tomography

The exposed Stage-3 pairs are replayed under intact rewriting, 50-percent
no-rewrite decay, exact static carry, and founder-clamped carry. Intact replay
must reproduce the retained Stage-3 outcomes. For each generation the run
records carrier reconstruction error, parent/child and founder-direction
cosines, between-history separation, within-history variance, fixed-form
potency in a fresh common garden, and target-free held-out A/B decoding.

Candidate raw carriers are collected from:

- strict windows 33--48, 33--64, 41--56, and 49--64; and
- developmental-overlap windows 17--32, 25--40, and 17--48.

One-generation common-garden potency ranks windows, with A/B carrier-direction
cosine and then stable identifier as tie breakers. The top two strict windows
and top overlap window continue. Ensemble bottlenecks of 1, 4, and 16 daughters
are diagnostics only and cannot earn a single-lineage verdict.

## Repair classes

Every repair receives only the daughter's raw carrier and frozen universal
parameters at runtime. It cannot receive the entering carrier, history label,
founder identity, target form, or phenotype prototype.

The simple class contains raw identity, fixed gains 0.5, 1, 2, and 4, and a
mean-zero gauge followed by the universal median founder L2 norm. The learned
class contains scalar affine, binwise diagonal ridge, and reduced-rank ridge
maps with ranks 8, 16, 32, and 64. Ridge candidates are 1e-4, 1e-3, 1e-2,
1e-1, 1, and 10.

Learned maps predict the entering parent carrier from the raw daughter carrier
using generations 1--4 of the exposed pairs. Hyperparameters use deterministic
eight-fold pair-grouped cross-validation. Normalized reconstruction error is
primary, A/B direction cosine is the tie breaker, and stable identifier is the
final tie breaker. Phenotype labels do not enter fitting. A final universal map
is refit on all exposed development pairs before selection.

## Selection and qualification

All retained timing/repair combinations run on the fresh 64-pair selection
cohort. A strict candidate is screen-eligible only if mean fixed-form crossover
is at least 0.20 at generation 4, 0.15 at generation 8, and 0.10 at generation
16; the generation-8 and generation-16 lower bounds are positive; survival is
at least 0.90; both directions are positive; at least half the pairs are
positive; and generation-8 advantage over 50-percent no-rewrite is at least
0.10 with a positive lower bound.

At most one simple and one learned strict candidate are retained. Within each
class, candidates rank by the minimum of generation-4, generation-8, and
generation-16 crossover, then generation-16 lower bound, then lower model
complexity, then identifier. The top overlap candidate is reported separately
but cannot enter strict confirmation.

Each provisional strict winner receives the complete Stage-3 causal ladder on
the selection cohort: zero every boundary, address shuffle, read disabled,
founder write disabled, no rewrite, ablation entering generation 3,
same-history and opposite-history contemporaneous rescue entering generation
4, opposite founder, and one-percent sign corruption. Qualification retains
the original Stage-3 gates, including 70-percent no-rewrite and ablation loss,
same rescue, opposite reversal, terminal form, survival, and corruption. If no
candidate qualifies, the campaign stops and the 96-pair confirmation cohort is
not simulated.

## Confirmation and verdicts

After review, each qualifying simple or learned mechanism is frozen and run on
the same 96 matched confirmation pairs. Each mechanism class receives alpha
0.0125 so the two-class familywise alpha is at most 0.025. A
`STRICT_RENEWED_CA_PLASTIC_HEREDITY` verdict requires every registered
ancestral-form and causal-renewal gate through generation 16. Active rewriting
must outperform the stale founder carrier.

The secondary observer uses deterministic disjoint replicate halves and
nearest-centroid decoding within each pair. It separately decodes the carrier
and a frozen visible descriptor combining accumulated 2x2 texture, terminal
2x2 texture, and the independent texture descriptor. Generation-16 balanced
accuracy must be at least 0.65 with lower bound above 0.55, exceed no-rewrite
and read-disabled by at least 0.10, fall to at most 0.55 after ablation, recover
at least 70 percent of excess-over-chance after same-history rescue, and reverse
under opposite rescue.

If carrier and phenotype pass while strict form fails, the secondary verdict
is `EXPRESSED_DRIFTED_LINEAGE_HEREDITY`. If only the carrier passes it is
`CRYPTIC_RENEWED_CARRIER_MEMORY`; otherwise it is `NO_DURABLE_RENEWAL`.
Secondary verdicts do not unlock the original compression/locality queue.

## Execution and artifacts

Each invocation uses at most 20 workers, stops new scientific submissions with
30 minutes reserved, and has a hard eight-hour wall budget. Checkpoints are
atomic and hash-bound to protocol, implementation, ancestors, cohorts, repair
models, and phase. Resume refuses a changed design.

The preconfirmation invocation produces `DIAGNOSTIC.json`, compressed
non-pickled carrier traces with manifests, `REPAIR_MODELS.npz`, `SCREEN.json`,
`SELECTION_DECISION.json`, `CONFIRMATION_DESIGN.json`, reports, queue, and
pollable status. It ends in `awaiting_human_review` or
`no_candidate_for_confirmation`; it never launches confirmation automatically.
