# Preregistered Life carrier falsification and discovery campaign

Date frozen: 2026-08-21, before generating any round-2 trajectory.

## Purpose and evidence boundary

The completed causal-heredity campaign found no general spatial carrier, but
Life rule `125398` (`B35678/S124678`) had a multiplicity-corrected exploratory
square-fragment advantage.  Its acquired donors were 253--256 live cells on a
16x16 torus and their ancestors often recovered the same observer form.  This
campaign first tests whether that lead is acquired-state information or generic
nucleation by a dense block.  It then asks a stronger question in rules with at
least two forms: does changing only the form-specific parental transplant alter
which form an identical recipient expresses?

The validated clean-room Life engine and its disclosed activity clock are
reused.  Sibling implementation code remains excluded.  The frozen 1,024-rule
atlas is a development catalogue only; all donors, futures, and holdouts use a
new namespace.  Screening chooses candidates but cannot establish a claim.

## Frozen panels and cohorts

The saturation audit contains rules 125398 and 124375.  Reference acquisition
targets 128 fresh strict donors per rule and each audit condition receives 128
futures for 48 generations.

The multi-form screen is the first 24 rules, sorted by decreasing strict rate,
then decreasing library size, then increasing rule id, among frozen-atlas rows
with library size at least two, strict rate in `[0.005, 0.5]`, and mean survival
at least 16.  A donor is eligible only when its established terminal density is
in `[0.05, 0.95]`.  Its form id is the registered 0.75-mass support of its
established primary centroid.  The two most frequent distinct forms are paired
within launch by minimum density difference, with a maximum difference of 0.05
and no donor reuse.  Reference screening requires 16 pairs and examines at most
32,768 fresh lineages per rule.

The screen uses 32 futures per arm for 16 generations.  At most four rules are
sealed for confirmation: their symmetric A/B crossover must be at least 0.10,
survival at least 0.90, both auxiliary observers must agree in direction, and
at least 12 pairs must be usable.  Ties use decreasing crossover then ascending
rule id.  The child-selection manifest is written before any holdout donor or
trajectory exists.

Each sealed rule then acquires 64 new pairs under a disjoint holdout namespace.
Every holdout arm receives 128 futures for 48 generations.  Mapping uses at
most 16 holdout pairs per rule and 32 futures per arm.

## Interventions

All form-specific comparisons use an identical frozen launch board as the
common recipient.  A transplant replaces every bit under a deterministic mask;
it is not OR-composited.  Square doses are `1/16, 1/8, 1/4, 1/2, 1`; strip and
two-lobe geometries are tested at one half.  Controls include exact-live-count
random states, block shuffles, an unrelated donor of the same form, and a
morphology surrogate.  A morphology surrogate preserves live count exactly
and must reach normalized live-neighbour-count error at most 0.02 and component
spectrum cosine at least 0.95 within 100,000 deterministic swap proposals.
Failure remains missing and is never reseeded.

The saturation audit additionally compares the historical square fragment
with intact and ancestor states, a generic all-live square, translation,
rotation, reflection, block shuffle, density random, morphology surrogate, and
empty, launch, and density-matched recipient backgrounds.

For confirmed candidates, carrier mapping includes a 4x4 tile deletion and
sufficiency scan, deterministic recursive quadrant shrinking, translations,
rotations, reflections, reciprocal A/B transplants, both-daughter pedigrees to
depth eight, process/copy noise multipliers 0/1/2, a 32x32 scale check, and all
17 one-bit B/S neighbours.  B0 is outside the encoded rule family, so the
neighbour count is 17 rather than 18.

## Outcomes and inference

Assignment to A or B requires cosine similarity at least 0.90 and a
best-versus-runner-up margin at least 0.05.  Missing, dead, and unresolved
futures remain in the denominator.  The primary score is

`min(P(A|A donor)-P(A|B donor), P(B|B donor)-P(B|A donor))`.

The matched donor pair is the independent unit.  Reference inference uses
10,000 deterministic pair-cluster bootstrap draws and paired permutation
tests.  The holdout family is Holm corrected across sealed candidates and
registered controls.

The historical saturation lead replicates only when square minus density
random is at least 0.08 with a positive 95% lower bound.  It is acquired-form
specific only if it also beats generic all-live, morphology, and ancestor
controls by at least 0.10 with positive corrected lower bounds.  Otherwise a
replicated lead is `SATURATION_NUCLEATION_ONLY`.

A `DURABLE_CAUSAL_CARRIER` requires generation-48 symmetric crossover at least
0.15 with positive corrected lower bound; positive A and B directions; at
least 0.10 advantage over random, shuffled, and morphology controls; target
probability at least 0.25; improvement in at least half of donor pairs;
positive depth-eight crossover; agreement by an independent observer; and the
same direction at 32x32 and under translation/rotation.  A generation-16 lead
that fails generation 48 or pedigree persistence is
`FORM_SPECIFIC_TRANSIENT_CARRIER`.  Otherwise the result is
`NO_CAUSAL_CARRIER_FOUND`.  Noise and neighbouring-rule results explain a
carrier but cannot rescue a failed primary gate.

## Reproducibility and execution

Profiles `smoke`, `pilot`, and `reference` differ only in sample size and panel
limits.  Every stage is atomically checkpointed under the complete design
digest.  Detached runs write `RUN.pid`, `run.log`, and atomic `STATUS.json`.
A 48-hour deadline completes in-flight checkpoints, emits an explicitly
partial result, and remains exactly resumable.  `COMPLETE` is written only
after adjudication of every scheduled stage.
