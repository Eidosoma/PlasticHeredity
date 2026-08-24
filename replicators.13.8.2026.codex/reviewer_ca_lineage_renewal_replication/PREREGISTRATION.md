# Preregistered strict CA lineage-renewal replication

Date frozen: 2026-08-23 UTC, before any fresh lineage outcome in this package.

## Question and evidence boundary

Can the locally replicated Rule-31649 motif channel close a causal,
self-renewing inheritance loop for 16 visibly reset generations? NewIdeas was
consulted only through an explicit allowlist of JSON data and Markdown
documents. Its outcome suggested this hypothesis and fixed the mechanism; its
result is not evidence in this replication. No source implementation was read,
hashed, imported, or executed.

The substrate is a 16 by 16 toroidal binary cellular automaton under rule 31649
(`B13456/S0578`). The claim is limited to synthetic CA lineage memory. It does
not imply metabolism, agency, biological life, or memory outside the total CA
state.

## Fixed lifecycle

The founder's 512-entry carrier is its Jeffreys-smoothed 3x3 motif log-frequency
difference from the frozen, label-blind local Stage-1 calibration. It is clipped
to [-4, 4]. Each generation starts from a pair-specific 50% neutral board that
is bitwise identical across A/B histories and across every generation. The CA
runs 64 sweeps with process-noise probability 0.002.

During sweeps 1--32, the fixed `motif_energy512-w32-s025-d32` reader accepts
carrier-energy-improving synchronous cell flips with probability 0.25. Reading
then stops. The daughter writer counts 3x3 motifs only during sweeps 49--64,
converts them using the same frozen reference and clipping rule, and multiplies
the raw daughter carrier by the universal scalar gain 0.5. This repaired
daughter-written carrier replaces the inherited carrier at the next boundary.
The primary observer is accumulated live 2x2 texture during sweeps 57--64; the
terminal 2x2 observer is an independent gate.

A dead branch cannot reproduce and remains in all denominators. Random fields
are semantically paired across A/B histories and every intervention. Neither
the writer nor gain repair receives a form label, parent, target, or outcome at
runtime.

## Cohort and interventions

All local Stage-1/2 donors and every donor exposed in the NewIdeas Stage-3/3R
data are excluded. The remaining frozen pool contains exactly 98 eligible,
nonreusing, same-launch A/B pairs within a density-difference caliper of 0.02.
Hash ordering assigns two pairs to a non-evidential engineering quarantine and
96 untouched pairs to confirmation. Confirmation uses 64 futures per history
and 16 generations.

Every confirmation pair receives exactly eleven registered conditions:
intact; zero at every boundary; address shuffle at every boundary; reading
disabled; founder writing disabled; no daughter rewrite with 50% inherited
amplitude retention; ablation entering generation 3; that ablation followed by
same-history or opposite-history intact-sister rescue entering generation 4;
opposite founder entering generation 1; and 1% carrier-sign corruption at every
boundary. No additional scientific experiment is included.

## Frozen inference

The matched founder pair is the independent unit. Outcomes are retained after
generations 1, 2, 4, 8, and 16. Intervals use 10,000 deterministic pair-cluster
bootstrap draws at alpha 0.0125. A strict pass requires all of the following:

- intact mean pair crossover at least 0.20 at generation 4, 0.15 at generation
  8, and 0.10 at generation 16, each with a positive lower bound;
- both generation-8 directions positive, at least half of pairs positive,
  survival at least 0.90 through generations 8 and 16, and generation-8
  terminal crossover at least 0.10 with positive lower bound;
- generation-8 intact advantages of at least 0.10 with positive lower bounds
  over zero, shuffle, reader-off, and founder-writer-off controls;
- at least 70% loss without daughter rewriting and an active-rewrite advantage
  of at least 0.10 with positive lower bound at generation 8;
- at least 70% loss after ablation by generation 4;
- at least 70% same-history rescue restoration and rescue advantage of at least
  0.10 with positive lower bound;
- opposite rescue at generation 4 and opposite founder at generation 8 each at
  most -0.10 with a negative upper bound; and
- generation-8 crossover at least 0.10 with positive lower bound under repeated
  1% sign corruption.

Passing every gate yields `STRICT_RENEWED_CA_PLASTIC_HEREDITY`. Persistence
without the registered rewriting loss is `STATIC_HIDDEN_TEMPLATE`. Passing at
generation 8 but failing durability at generation 16 is
`TRANSIENT_LINEAGE_MEMORY`. Incomplete data cannot pass.

The source's secondary drifting-lineage decoder is not adjudicated because its
independent texture descriptor is not operationally specified in the permitted
data/docs. It is explicitly non-gating for this strict original-form result.

Registration is hash-bound to the snapshot, upstream local artifacts, source
data/docs, cohorts, contract, local implementation, and sealed scalar
dependencies. Registration does not launch fresh outcomes. Checkpoints are
atomic and resumable; semantic seeds do not depend on worker count or finish
order. No later stage or added experiment starts automatically.
