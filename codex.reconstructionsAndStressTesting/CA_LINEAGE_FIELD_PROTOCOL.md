# Preregistered CA lineage-field round 4

Date frozen: 2026-08-22. This protocol is written before any lineage-field
outcome trajectory is generated. Round-3 CA artifacts are development inputs
that define fixed stimuli and continuous form targets. No Wagner implementation
source, tests, parameters, or data structures are inputs to this experiment.

## Question and clean-room boundary

The experiment asks whether a mature Life-like CA form can write a separate,
slow local field which crosses a reproduction boundary while the visible board
is reset bit-for-bit, biases the daughter's early development, and must be
rewritten by that daughter for the effect to persist. A positive result is a
synthetic two-timescale carrier mechanism. The slow field remains part of the
total CA state; no result is evidence for memory outside that state, metabolism,
agency, or biological life.

Only the frozen Rule-31649 narrow prototypes and the Rule-31648 and Rule-70366
round-3 holdout pairs are used. Pair selection is deterministic and sealed
before round-4 outcomes. All outcome RNG namespaces are new.

## Two equally weighted mechanisms

The visible state `X` is a periodic 16 by 16 binary lattice. The carrier `M` is
a signed 16 by 16 field clipped to [-1, 1]. Both mechanisms use the same generic
reader. After the ordinary Life-like update, positive `M` can turn a predicted
dead site on and negative `M` can turn a predicted live site off, with
probability `kappa * abs(M)`. The reader has no form label or prototype.

The latch writer takes per-site occupancy over its write window. Occupancy at
or above a symmetric upper threshold writes +1, occupancy at or below the lower
threshold writes -1, and intermediate occupancy retains the decayed latch.

The diffusing writer updates on each write sweep as

`M <- clip((1-D-W) M + D mean8(M) + W (2X-1), -1, 1)`.

Outcome-blind calibration considers `kappa` in {0.025, 0.05, 0.10}, decay in
{0.40, 0.55, 0.70}, latch thresholds (0.60, 0.40) and (0.70, 0.30), diffusion
in {0.04, 0.08}, and write gain in {0.08, 0.12}. Calibration sees generic
synthetic boards only. It uses viability, perturbation, saturation, and decay,
never A/B assignments. One configuration per mechanism is sealed before main
outcomes. Both mechanisms receive identical main-run resources.

## Lineage lifecycle

Every founder carrier starts at zero. Its mature A or B board runs for 16
writer-only sweeps. At the reproduction boundary, only the carrier is retained.
Every descendant begins from the matched pair's identical neutral launch board.
The reset checksum is asserted before every generation.

Each descendant generation has 32 sweeps. The carrier is read during sweeps
1--8, the base CA develops without reading during sweeps 9--32, the daughter
writes during sweeps 17--32, and phenotype observations accumulate over sweeps
25--32. Process noise 0.002 is applied after carrier reading. Semantic RNG
streams are paired across histories and interventions. Carrier decay and the
registered bottleneck are applied after writing. Lineages are assessed after
generations 1, 2, 4, 8, and 16. Empty daughters die; nonempty unresolved forms
remain in the denominator and may reproduce.

## Intervention matrix

Both mechanisms and all three rules receive: intact, zero at every boundary,
spatial shuffle, read-disabled, founder-write-disabled, no rewrite after the
founder, ablation after generation 2, same-history rescue entering generation
4, opposite-history rescue entering generation 4, and opposite founder-carrier
transfer. Rescue uses a contemporaneous intact sister branch and never feeds
back into it.

Rule 31649 additionally receives 2x2, 4x4, 8x8, and global block averaging;
matched random-coordinate retention of 64, 16, and 4 values; 1% and 5% carrier
sign corruption; and visible-state controls carrying 64 or 16 terminal bits
through an intentionally incomplete reset.

## Sampling, compute, and inference

The preferred reference profile uses 32 pairs and 64 futures for Rule 31649,
24 pairs and 32 futures for each holdout, and 32 pairs by 32 futures for the
diagnostics. A label-free timing benchmark may select one of two sealed smaller
profiles, symmetrically for both mechanisms, if the preferred profile is not
projected to finish safely. The pair is the independent unit. Missing, dead,
and unresolved futures stay in denominators. Pair-cluster bootstraps use 10,000
draws in the reference profile, with correction across mechanism and holdout
families.

The primary form observer is accumulated live 2x2 texture with cosine at least
0.90 and an A/B margin at least 0.05. Terminal 2x2 texture and component geometry
are independent observers. Symmetric crossover is the smaller of the A-history
and B-history directional probability differences.

`RENEWED_LINEAGE_CARRIER` requires intact crossover >=0.15 at generation 8 and
>=0.10 at generation 16 with positive confidence bounds, both directions
positive, at least half of pairs positive, survival >=0.90, advantages >=0.10
with positive bounds over zero/shuffle/read-disabled/write-disabled, loss of at
least 70% without rewrite and after ablation, restoration of at least 70% by
same-history rescue, opposite-history reversal <=-0.10, and a positive
independent local observer. Compression and cross-rule successes add the
registered `COMPRESSED_` and `CROSS_RULE_` tiers. A causal effect that does not
need rewriting is labelled `STATIC_HIDDEN_TEMPLATE`. An incomplete stage can
never receive a positive verdict.

The detached run uses 20 workers. It stops submitting scientific checkpoints
at 7 h 30 min, reserves time for adjudication and shutdown, and has a hard
8-hour monotonic wall deadline. Checkpoints are atomic and resumable. The output
must contain the protocol/design/input/code hashes, selected timing profile and
parameters, cohort IDs, status and ETA, complete statistics, a scientific
report, and a lay summary.
