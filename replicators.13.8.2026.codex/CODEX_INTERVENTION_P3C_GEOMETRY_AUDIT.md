# P3c post-hoc geometry audit protocol

Status: additive post-hoc analysis of the already sealed P3b result.

This audit does not change, repair, or replace the P3b registration or result.
It uses only the checksum-verified P3b artifacts and computes no new simulated
future.  Its purpose is to explain why the P3b balanced-log random arm was not
equivalent to no intervention and to motivate a prospectively registered P3c
specificity control.

## Frozen source

- `results_intervention_replication/p3b_beta_surgery_dose_bridge`
- all source checksums must pass before analysis;
- all 80 matrices, both candidates, all six landmarks, both branch halves, and
  the four registered structural no-action singleton states are retained.

## Geometry

For composition fractions `x` and catalytic matrix `beta`, calculate for each
state and arm:

- occupied-block Frobenius norm;
- occupied-block sum, arithmetic mean, and geometric mean;
- catalytic throughput `T = x.T @ beta @ x`;
- Perron spectral radius of the occupied block;
- row- and column-strength dispersion;
- the perturbation's radial projection on the original occupied block.

All differences and log ratios are relative to the same state's NOOP geometry.
The four structural singleton states contribute exact zero shifts for every
arm, matching their sealed all-arm no-action handling.

## Outcome association

For each candidate and fixed branch half, use the five standard landmarks
`20, 35, 50, 65, 80` as the main descriptive scope and landmark 60 as an
external-compatibility scope.  Report:

1. state-centred ordinary-least-squares slopes of realised arm-minus-NOOP
   JOINT_BREAK_RUN3 probability on each geometry shift;
2. mean within-state Spearman association across arms;
3. arm-specific matrix-level geometry and outcome shifts;
4. 4,096 whole-matrix bootstrap draws for slopes and correlations.

No p-value or interval in this audit is confirmatory.  No geometry variable may
be selected by its outcome association for the P3c primary analysis.  P3c's
primary mediator is fixed in advance as `log(T_arm / T_NOOP)`.

## Claim boundary

This audit may explain an observed control-arm failure and justify a new
prospective control.  It cannot establish a causal mediator, rescue P3b's
failed formal specificity gate, or serve as confirmation evidence for P3c.
