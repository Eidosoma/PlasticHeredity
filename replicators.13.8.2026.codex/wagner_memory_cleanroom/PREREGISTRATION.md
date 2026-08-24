# Preregistration: Wagner memory stack v1

Status: implementation-frozen scientific contract. The canonical numerical
contract is `protocols/wagner-memory-v1.json`.

## Scope and claim hierarchy

This campaign tests memory in N=10 sequential Wagner networks. It does not test
F12 prediction, Boolean networks, evolution, recipient portability, or a newly
discovered carrier. The primary claims form a ladder:

1. a complete expression state carries acquired A/B history;
2. a separately represented hysteretic carrier is naturally written from that
   history and renews it across complete expression resets;
3. read/write, reversal, ablation, and rescue controls make that carrier causal;
4. bottlenecks test whether the effect is distributed.

Hard state transfer and the full carrier are primary. Soft writing, the noise
boundary, a decaying-mark screen, and smaller bottlenecks are separately
reported secondary results. No tier substitutes for another.

## Independent Wagner contract

Each rulebook has ten binary genes and one fixed weighted regulatory matrix.
Two seed-addressed, separated fixed points are embedded in a sparse random
matrix and then verified under deterministic sequential updates. Eligibility
uses only this pre-assay verification and a frozen basin test. All proposed and
accepted rulebooks are recorded.

One sweep updates genes 0 through 9 sequentially. The next value is the sign of
the weighted input plus any registered cue/carrier field and regulatory noise;
an exact zero retains the preceding value. Regulatory noise is zero-mean normal
with variance `theta`. A post-sweep expression flip probability supplies the
registered stochastic future variation. Targets, neutral midpoint states,
fields, challenges, and random streams are deterministic functions of semantic
coordinates.

## Expression-state assay

The state confirmation uses 240 fresh rulebooks. Hard writing clamps every gene
to the history target for one sweep. Soft writing applies a target-aligned field
of strength 0.5 for one sweep. State arms are self continuation, exact state
transplant, reset, destination-matched, descriptor-matched, and pattern shuffle.
Release, 20% neutral damage, and a forced break are assayed at descendant ages
0, 1, 2, 4, and 8. The retained design has exactly 6,389,760 futures.

Minimal state heredity requires acquisition of both histories, correct strict-8
retention, a state-minus-reset risk gain of at least 0.05, A/B crossover of at
least 0.05, held-out log-loss gain of at least 0.02 nats, split-half reliability
of at least 0.80, state/shuffle separation, and positive age-one effects. All
registered lower bounds must be above zero.

## Noise and simple slow-mark boundary

Separate 96-rulebook cohorts evaluate hard and soft writers at theta 0, .0025,
.005, .01, .03, and .10. A negative rulebook-level slope and positive theta-0
minus theta-.10 lower bound define the boundary result. The decaying-mark screen
crosses half-lives 1, 4, 8, and 16 with couplings .25 and .50. No passing setting
is reported as `NO_SLOW_MARK_CONFIRMED`, not as proof that slow marks cannot
exist.

## Lineage-carrier assay

The untouched carrier cohort contains 240 new rulebooks and 32 futures per
future half. The fixed candidate is a ten-entry signed latch with write
threshold one, sixteen developmental-cycle retention, and coupling one. A
founder cue writes the carrier from expression. Every child begins from the
same neutral expression state, reads only its inherited carrier, develops for
four sweeps, rewrites the carrier, and transmits only the carrier. Parental
expression is never copied.

Checkpoints are generations 0, 1, 2, 4, 8, and 16. Arms are natural full,
targeted and matched-random k=5/3/1, shuffle, opposite history, zero,
write-disabled, read-disabled, no-rewrite, generation-2 ablation,
generation-2 ablation plus generation-3 rescue, and an exact-write ceiling.
Challenges are release, neutral damage, and forced break. Targeted and random
k=10 are deliberately absent because both are identical to the full carrier.

The carrier primary requires generation-4 risk gain and crossover of at least
.05, held-out log-loss gain of at least .02 nats, positive simultaneous lower
bounds, reliability at least .80, and positive generation-8 and generation-16
effects under all challenges. Causality additionally requires loss under the
registered negative controls, reversal under opposite history, at least 70%
loss after ablation, and at least 70% restoration after rescue. Distribution is
supported when k=5 retains less than 70% of the full effect.

## Inference and operations

The rulebook is the independent unit. Futures are paired across arms and split
into two frozen halves. Inference uses 4,096 whole-rulebook bootstraps and 95%
simultaneous intervals. Every outcome remains in its denominator. Confirmation
is regenerated independently and a seed-frozen replay sample must be exact.

The two-GPU admission benchmark uses discarded seeds. It must project the full
campaign below 10.5 hours after a 25% margin. Scientific execution requires
exactly two CUDA workers, stops new shards after 11 hours, starts sealing by
11.5 hours, and has a 12-hour hard deadline. No gate changes after launch.

