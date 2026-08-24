# Preregistration: corrected Wagner memory stack v2

Status: frozen before scientific execution. The canonical numeric contract is
`protocols/wagner-memory-v2.json`. V1 is not reanalysed or overwritten.

## Source ensemble

Each cohort uses fresh N=10 sequential Wagner rulebooks. A proposal is a dense,
float64 zero-bias Gaussian regulatory matrix with its diagonal retained. All 1,024
binary states are enumerated, deterministic successors and attractor basins are
resolved, and the largest eligible complementary point-attractor pair is chosen.
Each basin must contain at least 5% of states. Two complementary midpoint starts
are frozen exactly five bits from either target. A forced-break state is the
closest state outside the requested target basin. Every proposal digest and
eligibility decision, plus every accepted matrix and landscape summary, is
retained.

Smoke, quick, full, and admission-benchmark source domains are cryptographically
disjoint. State and carrier audit stages reuse only their corresponding frozen
profile/stage domain. Thus diagnostics and runtime admission never consume a
scientific full-cohort rulebook before its registered stage.

Before each developmental cycle, expression coordinates flip independently with
probability .05. A cycle then develops under the sequential Wagner rule to a
point/cycle endpoint or the 100-sweep guard. Two endpoints are kept separate.
Strict retention is exact occupation of a target point form for eight
consecutive adult cycles among 32. Prediction uses the first deterministic
point attractor occupied for three consecutive adult cycles, pooled into the
registered A/B/other destination categories. A/B labels, midpoint IDs, future
halves, and all stochastic coordinates are explicit. `futures_per_cell` is the
total count and is divided equally between the two halves.

## Expression state and slow mark

The state cohort has 240 rulebooks and exactly 6,389,760 futures. The hard writer
is a one-cycle full persistent clamp. The soft writer is a one-cycle target field
of strength .5 times each unperturbed regulatory-row norm. State, reset, matched,
shuffle, recurrence, ages 1/2/4/8, and all registered challenges follow the
retained design. Confirmation includes acquisition, absolute strict-8 hold,
70%-of-injection-ceiling retention, risk, direct A/B crossover, held-out history
log loss, split-half crossover reliability, shuffle, pathwise identity, and
generation-one gates.

The 96-source slow-mark stage has exactly 3,932,160 futures. A hard-written adult
trajectory writes a signed mark using `m_next = rho*m + (1-rho)*adult`. Each
cycle the mark stochastically biases its matching newborn coordinate and is then
updated from the realized adult. Washout age advances donor expression and its
mark together under those recurrent read/write rules; the mark is never held
frozen while expression alone ages. The screen and mechanism matrix include mark-only,
state-plus-mark, reset, fixed shuffle, disabled writing, inert reading, targeted
and random ablation, and rescue at ages zero and eight. A candidate must pass the
history gates, beat every registered negative control, lose more under targeted
than random ablation, and be restored by rescue.

## Renewable lineage carrier

The untouched Wagner-only cohort has 240 new rulebooks, two midpoints, 32 futures
per cell, checkpoints 0/1/2/4/8/16, three challenges, and all sixteen discovery
and mechanism arms: 8,847,360 futures. This is a prospective full Wagner
confirmation; the archived 240-Wagner/96-Boolean full confirmation was never run.

The founder starts at its registered midpoint. The selected full, dwell-one
persistent clamp creates an expression trajectory; a threshold-one hysteretic
latch is written from that trajectory. Direct carrier assignment occurs only in
the explicitly labelled exact-write ceiling. Every child starts from the same
midpoint. During each of four adult developmental cycles, every still-live
latch coordinate can overwrite its matching noisy newborn-expression coordinate;
the realized adult then renews or rewrites the latch. Only the bottlenecked
latch is transmitted. Checkpoint challenges branch without feeding back. During
a checkpoint assay the frozen latch is read while its per-coordinate TTL remains
positive, is not rewritten, and is discarded after the 32-cycle branch.

The latch retains an entry for sixteen adult developmental cycles. Matching expression
renews it; an opposing value can replace it only after expiry and after the
registered consecutive-write threshold. Influence is carrier magnitude times an
exact one-sweep recipient sensitivity. Targeted and matched-random arms share all
future stochastic coordinates.

Primary gates are generation-4 risk gain >=.05, direct A/B crossover >=.05,
held-out history log-loss gain >=.02 nats, adjusted lower bounds above zero,
split-half crossover reliability >=.80, and positive generation-8/16 effects for
every challenge. Causality requires registered control separation, opposite
reversal, >=70% ablation loss, and >=70% rescue. Bottleneck results remain
separate.

## Inference and integrity

The rulebook is the independent unit. A single whole-rulebook bootstrap resample
is shared across each family; one-sided simultaneous lower bounds use the 95th
percentile of the family-wise maximum bootstrap shortfall. History decoding is
trained per rulebook on one future half and evaluated on the other from absolute
A/B/other destinations against that rulebook's history-pooled baseline, then
the halves are reversed. Split-half reliability is the Pearson correlation of source-level
direct crossover, with exact equality defined as one.

Every aggregate cell retains its ordered future-outcome digest. Future IDs encode
stage, source, midpoint, history, condition, half, and within-half index. State
and carrier are regenerated in independent audit stages; verification compares
all cell digests and separately compares outcomes for registered replay IDs.
Expected source, cell, and future counts are hard gates. Source and environment
hashes are sealed.

Two CUDA workers are required. Admission includes a 25% margin and must project
below 10.5 hours. No stage starts after hour 11, sealing starts by 11.5 hours, and
the process has a hard 12-hour deadline.
