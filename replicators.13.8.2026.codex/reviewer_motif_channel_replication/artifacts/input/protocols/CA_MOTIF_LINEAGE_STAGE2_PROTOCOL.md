# Preregistered CA motif-lineage Stage 2

Date frozen: 2026-08-22. This protocol is frozen before any Stage-2 daughter
trajectory is generated. The user explicitly authorized continuation after
reviewing the completed Stage-1 result.

## Question and frozen mechanism

Stage 2 asks whether the Stage-1 motif-energy result is a reusable inherited
form channel or a controller tied to its original pairs and reset board. The
reader is imported verbatim from `STAGE_DECISION.json`:

- carrier: 512 signed 3 by 3 motif energies;
- parent write window: 32 sweeps;
- read strength: 0.25;
- daughter read duration: 32 sweeps;
- registered outcome checkpoint: sweep 64.

No parameter, threshold, carrier address, or checkpoint may be tuned using a
Stage-2 outcome. A positive result remains a one-generation form channel, not
renewed Plastic Heredity.

The clean-room exclusion continues: no Wagner or Fable implementation source
is read, imported, hashed, or executed.

## Fresh cohort and reset panel

All 176 pair IDs used for Stage-1 calibration, discovery, or validation are
excluded. Two additional IDs used to verify the Stage-2 smoke runner are
permanently quarantined as development-only before reference outcomes are
examined: `narrow-0468-life-31649-2-1381-life-31649-2-1497` and
`narrow-0759-life-31649-3-528-life-31649-3-91`. The reference cohort is the
first 96 of the remaining 704 pairs under the new
`plastic-ca-motif-lineage-stage2-v1` hash ordering. Each receives 64 paired
futures per history. Every A/B comparison has an asserted identical visible
reset and paired semantic randomness.

The primary environments are the pair's native reset; each of the four frozen
round-3 launch resets; a native reset translated by (3,5); a covariant 90-degree
rotation; and a covariant horizontal reflection. Rotation/reflection permute
the 512 carrier addresses consistently, and observations are inverse-transformed
before scoring. Random boards with exactly 10%, 30%, and 50% live cells are
registered density stresses.

## Causal and transfer panel

Every primary environment receives the intact carrier, zero carrier,
read-disabled carrier, one common address shuffle, independently permuted
value-matched decoys, opposite-history transfer, an unrelated pair's
same-history carrier, and the A/B midpoint carrier. The native reset additionally
repeats process noise 0.002 and one-percent carrier-sign corruption. Density
stresses receive intact, zero, opposite-history, and unrelated-pair transfer.

The unrelated source is the next pair in the sealed cohort, cyclically; no
outcome determines source matching. The midpoint gives both histories exactly
the same carrier and must produce no directional effect under paired randomness.

A separate native-reset dose test uses carrier contrasts 0, 0.25, 0.50, 0.75,
and 1. At contrast zero both histories receive their midpoint. At contrast one
they receive the intact A and B carriers. The form effect must rise monotonically
within tolerance 0.03, have rank correlation at least 0.90, and have a positive
pair-bootstrap slope of at least 0.10.

## Writer audit and outcome gates

Before daughter outcomes, 32 new pairs audit the writer. Raw parent motif
frequencies must be translation-, rotation-, and reflection-equivariant to
maximum absolute error 1e-6. An adjudication-only leave-one-pair-out nearest
centroid classifier must identify A/B carrier histories with at least 80%
accuracy. Labels used for this audit never select or alter the reader.

The primary observer remains trailing-eight-sweep accumulated live 2x2
texture. Terminal 2x2, component geometry, occupancy, autocorrelation, and
low-frequency power are independent observers. Dead and unresolved futures
remain in denominators. Pair-cluster bootstrap intervals use 10,000 draws and
familywise correction across primary environments.

Each primary environment must have intact crossover at least 0.15 with a
positive lower bound, positive directions, survival at least 0.90, terminal
crossover at least 0.10, advantages at least 0.10 over zero, read-disabled,
shuffle, and matched-random controls, opposite-history crossover at most -0.10
with a negative upper bound, unrelated transfer at least 0.10 and at least 70%
of intact, and midpoint magnitude at most 0.02. All primary environments, the
writer audit, dose response, and native noise tests must pass for
`GENERAL_REUSABLE_MOTIF_CHANNEL`. Passing all density stresses adds the
`DENSITY_ROBUST_` tier. An incomplete run cannot pass.

## Detached gate

The reference run uses 20 workers, stops submitting work at 7 hours 30 minutes,
and has a hard eight-hour deadline. Checkpoints are atomic and bound to all
code, protocol, cohort, Stage-1, reset, and configuration hashes. `STATUS.json`
is the polling interface and `--resume` accepts only the identical design.

The run freezes `RESULTS.json`, `REPORT.md`, `LAY_SUMMARY.md`,
`STAGE_DECISION.json`, and `QUEUE.json`. Stage 3 cannot launch automatically.
Only a reviewed Stage-2 pass may advance to the multigenerational writing,
ablation, and rescue experiment.
