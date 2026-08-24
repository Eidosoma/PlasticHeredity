# Preregistered CA motif-lineage Stage 1

Date frozen: 2026-08-22. This protocol is frozen before any Stage-1 outcome
trajectory is generated.

## Question and evidence boundary

The experiment asks whether Rule 31649's A-like and B-like forms can write a
translation-equivariant motif carrier which makes bitwise-identical reset
daughters develop differently. A positive result is a one-generation
controllability upper bound. It is not evidence of renewed Plastic Heredity;
daughter rewriting and multigenerational causal interventions are reserved for
Stage 3.

This is clean-room work. Frozen CA artifacts and our CA implementation are the
only executable inputs. No Wagner or Fable implementation source may be read,
imported, hashed, or executed.

## Disjoint cohorts

Rule-31649 pairs from the frozen round-3 acquisition are ordered with the new
`plastic-ca-motif-lineage-stage1-v1` hash namespace. The reference profile
assigns 64 pairs to label-blind calibration, the next 48 to discovery, and the
next 64 to untouched validation. Pair identities cannot overlap. Discovery
selects parameters; validation outcomes cannot affect selection.

Every A/B daughter comparison begins from the same neutral 16 by 16 board and
uses paired semantic random streams. A checksum assertion precedes every run.

## Two-level local ceiling

Both writers observe the post-update states of a mature parent for 16 or 32
sweeps. A label-blind reference table is pooled over both histories in the
calibration cohort with Jeffreys smoothing.

The `contextual256` carrier contains one signed mark for each eight-neighbour
ring. The mark is the parent's conditional centre-occupancy probability minus
the reference probability. A daughter cell reads only the entry addressed by
its current ring. Positive marks may activate predicted-dead cells and negative
marks may inhibit predicted-live cells. A zero table is exactly inert.

The `motif_energy512` carrier contains clipped log-frequency differences for
all binary 3 by 3 motifs. After the ordinary CA step, each cell compares the
carrier energy before and after flipping it across the nine motifs containing
that cell. All decisions are calculated from the same predicted board and
applied synchronously. The reader has no label, prototype, absolute coordinate,
or global target-distance objective.

## Discovery screen and frozen selection

The reference screen contains exactly 64 configurations: two carrier families,
write windows 16 and 32, read strengths 0.25, 0.50, 0.75, and 1.00, and read
durations 8, 16, 32, and 64. Each receives 48 discovery pairs and 16 paired
futures per history. Daughters are scored at sweeps 8, 16, 32, and 64.

Within each family, configurations are ranked by symmetric accumulated-2x2
crossover, survival, terminal-2x2 agreement, fraction of positive pairs, and
checkpoint stability. Ties prefer a shorter read, weaker reader, and shorter
write. The best two per family and their selected checkpoint are frozen before
validation.

## Untouched validation and controls

Each of four nominees receives 64 untouched pairs and 64 paired futures per
history. Conditions are intact carrier, zero carrier, reader disabled, shuffled
addresses, opposite-history transfer, unrelated same-form transfer, process
noise 0.002, one-percent sign corruption, the frozen round-4 spatial latch
benchmark, and a deliberately incomplete visible-64-bit reset assay control.

The primary observer is accumulated live 2x2 texture over the trailing eight
sweeps. Terminal 2x2 texture, connected-component geometry, occupancy,
autocorrelation, and low-frequency spatial power are independent diagnostics.
The inherited 3x3 statistic cannot be the primary outcome. Dead and unresolved
futures remain in denominators. Pair-cluster bootstrap intervals use 10,000
draws; the four nominee gates use familywise Bonferroni intervals.

`LOCAL_MOTIF_CONTROLLABILITY` requires intact symmetric crossover at least
0.15 with a positive lower interval bound, both directions positive, at least
half of pairs positive, survival at least 0.90, advantages at least 0.10 with
positive bounds over zero, shuffle, and reader-disabled controls, opposite
history crossover at most -0.10 with a negative upper bound, and a positive
independent observer. `ROBUST_LOCAL_MOTIF_CONTROLLABILITY` additionally
requires crossover at least 0.10 with a positive bound under process noise and
carrier corruption. An incomplete run cannot pass.

## Detached execution and five-stage queue

Stage 1 is an atomic, resumable detached campaign. It writes `RUN.pid`,
`run.log`, and atomic `STATUS.json`; stops submitting scientific checkpoints at
7 hours 30 minutes; and has a hard eight-hour wall deadline. Design, input,
code, cohort, and protocol hashes bind every checkpoint.

Stages 2--5 are queued as reader generalization, the renewed-heredity causal
ladder, compression/robustness, and strictly local inheritance. They never
launch automatically. Stage 1 freezes `RESULTS.json`, `REPORT.md`,
`LAY_SUMMARY.md`, `STAGE_DECISION.json`, and `QUEUE.json`. If no Stage-1 nominee
passes, the queue halts for a dual-attractor/rule-search redesign.
