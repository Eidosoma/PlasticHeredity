# Preregistered CA carrier round 3: continuous forms and observer scope

Date frozen: 2026-08-21.  This document is written before any round-3
trajectory is generated.  Round-1 and round-2 results are development data;
they may select hypotheses and calibrate outcome-blind controls, but they are
not confirmation observations.

## Scientific question and evidence ladder

Round 2 found a large, bidirectional local-texture crossover in Life-like rule
31649 (`B13456/S0578`).  The 16 retained pairs had continuous A/B target cosine
similarities from 0.393 to 0.782.  In contrast, pooling the discrete support IDs
2868 and 3892 produced nearly identical centroids (cosine 0.9956) and weak
classification.  Round 3 therefore never treats those support IDs as forms.
It distinguishes four claims:

1. a causal, pair-specific texture address;
2. a reusable local texture form defined by frozen continuous prototypes;
3. durable local Plastic Heredity, requiring acquisition and pedigree tests;
4. observer-robust Plastic Heredity, additionally requiring a global or
   mesoscale observer and morphology-balanced controls.

Survival or recovery without source-specific crossover is generic nucleation,
not heredity.  A state present before the strict break is an ancestor control;
an acquired-state claim requires the post-break donor to beat that control.

## Frozen narrow hypothesis

The prototype pair is selected without transplant outcomes: it is the retained
rule-31649 pair with minimum primary-target cosine, with pair ID as tie breaker.
Its continuous target vectors are sealed in `CALIBRATION.json`.  A fresh donor
matches prototype A or B only when cosine to the selected target is at least
0.95 and exceeds cosine to the other target by at least 0.05.  Pairs share a
launch, have terminal-density difference at most 0.02, and never reuse donors.
The retained primary target is the original eight-generation centroid.  The
round-2 checkpoint did not retain all eight terminal boards, so newly defined
secondary/global prototype targets are computed from the retained terminal
donor board; that limitation is explicit in the seal and those observers are
validators, never discovery selectors.

The reference acquisition examines exactly 32,768 lineages from each of four
launches.  The first 64 hash-ordered pairs are confirmation pairs, the next 32
are transfer pairs, and the next 32 are mapping pairs.  Fewer than 64 fresh
pairs is `UNDERPOWERED_ACQUISITION`; donors are never recycled to fill a cohort.
The 16 retained round-2 pairs are replayed only under new recipient and future
random streams.

Primary transplantation replaces bits under the mask; it is not OR
composition.  The reference confirmation uses a half-board square and 128
futures per arm through generation 64, with donor and ancestor arms extended
to generation 128.  Registered controls are pre-break ancestor, acquisition
anchor, exact-count random pixels, 2/4/8 block permutations, same-prototype and
opposite-prototype unrelated donors, all-live, and a conditional morphology
null.  Secondary arms cover doses 1/16, 1/8, 1/4, 1/2, 3/4, and 1; square,
strip, two-lobe, and dispersed geometry; translations, rotations, reflections;
zero/half/standard/double noise; 32 and 64 cell worlds; and all 17 one-bit
neighbours of rule 31649.  A split mapping cohort discovers a 4x4 tile mask on
16 pairs and tests it on 16 untouched pairs.  Both daughters are retained for
eight pedigree generations.

## Conditional morphology null

For every fragment, 4,096 deterministic exact-count rearrangements are
generated in the reference profile.  They are ranked by live-neighbour
histogram distance, connected-component spectrum cosine distance, and
low-frequency structure-factor distance.  The best 32 comprise the null
ensemble, so a missing control is impossible.  Balance calipers are the 90th
percentiles of best-achieved distances for the 32 historical rule-31649 source
fragments, rounded outward to 0.01.  Calibration uses no transplant outcome.
The null is always reported; the strongest claim additionally requires at
least 90% of fresh fragments to meet every frozen caliper.

## Broad search

The search covers all 256 ECA rules and all 1,024 rules in the frozen local
Life-like registry.  It is not a claim about all 131,072 encoded Life-like
rules.  Initial discovery examines 4,096 ECA and 2,048 Life lineages per rule,
retaining at most 256 certified switchers.  Continuous donor centroids are
partitioned by a deterministic, order-invariant complete-linkage-constrained
algorithm: donors are sorted by ID and may join a cluster only when cosine to
every member is at least 0.95.  Clusters need 16 donors and two launches.
Candidate cluster pairs need centroid cosine at most 0.80 and 16 non-reusing,
same-launch pairs with density difference at most 0.05.

Life discovery uses accumulated live 2x2 texture.  Held-out descriptions are
terminal 3x3 texture, orientation-resolved autocorrelation, connected-component
geometry, and low-frequency structure.  ECA discovery uses final-4 texture;
held-out descriptions are cyclic 5--8 neighbourhoods, run-length spectra,
low-frequency structure, and a figure/ground final-4 census derived from the
rule-specific domain dictionary over three consecutive terminal rows. Observer
vectors are normalized per generation before established-generation centroids
are taken.

The 64 highest feasibility-ranked Life rules and 32 ECA rules extend to 32,768
lineages.  At most 32 Life and 16 ECA candidates are screened with 16 pairs and
32 futures through generation 32.  At most eight per substrate enter a disjoint
fresh holdout with 64 pairs, 128 futures, and 64 generations.  Discovery,
extension, screen, and holdout use non-overlapping namespaces.  Selection is
deterministic and sealed before downstream data exist.

## Inference

The matched donor pair is the independent unit.  Missing, dead, and unresolved
futures stay in denominators.  Symmetric crossover is the smaller of
`P(A|A)-P(A|B)` and `P(B|B)-P(B|A)`.  Pair-cluster bootstrap confidence
intervals use 10,000 deterministic draws in the reference profile.  Wide
holdouts are Holm-corrected within substrate and claim family; the narrow
preregistered hypothesis is a separate family.

`CAUSAL_PAIR_TEXTURE_ADDRESS` requires generation-64 crossover at least 0.15,
a positive 95% lower bound, both directions positive, at least half of pairs
positive, survival at least 0.90, and at least 0.10 advantage with positive
lower bound over exact-count and block controls.  `REUSABLE_LOCAL_TEXTURE_FORM`
also requires fresh prototype-matched donors, a positive independent local
observer, and acquired-minus-ancestor advantage at least 0.10 with positive
lower bound.  `DURABLE_LOCAL_PLASTIC_HEREDITY` additionally requires the same
direction at generation 128 and through both-daughter depth eight.
`OBSERVER_ROBUST_PLASTIC_HEREDITY` additionally requires global/mesoscale
crossover at least 0.10 with a positive corrected lower bound, morphology
balance, and positive results under at least one geometry, scale, and moderate
noise change.  Lesser outcomes are explicitly labelled
`TRANSIENT_TEXTURE_ONLY`, `PAIR_SPECIFIC_ONLY`, `GENERIC_NUCLEATION_ONLY`,
`UNDERPOWERED_ACQUISITION`, or `NO_CAUSAL_CARRIER_FOUND`.

## Execution and reproducibility

Profiles change sample counts only.  Every task has a content-addressed atomic
checkpoint.  A detached reference run uses 20 workers, a 48-hour deadline,
single-threaded numerical libraries, and writes `RUN.pid`, `run.log`, and an
atomic `STATUS.json` with stage progress and ETA.  The deadline is checked only
between checkpoints.  `--resume` preserves design hashes, selections, and RNG
streams.  `COMPLETE` is written only after adjudication and artifact checks;
otherwise `PARTIAL` states exactly what remains.
