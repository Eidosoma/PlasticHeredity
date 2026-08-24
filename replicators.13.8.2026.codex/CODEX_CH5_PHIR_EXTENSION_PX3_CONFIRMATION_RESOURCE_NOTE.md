# PX3 confirmation resource note

Date: 2026-08-21

This note changes only the execution ceiling and worker limit for the still
unregistered PX3 confirmation. It does not change any scientific matrix,
state, arm, edit-selection rule, future count, endpoint, estimator, seed,
inference procedure, or gate.

## Timing

The note was written after the resource-bounded 12-matrix development result
was sealed and before confirmation registration or generation of any PX3
confirmation matrix. At this point the confirmation registration and
confirmation work directories did not exist.

## Evidence and correction

The original confirmation allocation of 12 aggregate process CPU-hours was a
wall-time estimate recorded in the CPU-time field. The completed development
generation used 48.2749 process CPU-hours for 12 matrices. Confirmation has
24 matrices but 0.24 times the future-simulation workload per matrix:

`(2 candidates * 3 landmarks * 4 arms * 64 branches) /
 (2 candidates * 2 replicates * 4 landmarks * 25 arms * 16 branches) = 0.24`.

This predicts about 23.17 process CPU-hours per confirmation pass and 46.34
process CPU-hours for generation plus exact replay, before overhead and
matrix-to-matrix runtime variation.

The corrected cumulative confirmation ceiling is therefore 64 process
CPU-hours. Confirmation runs detached with at most eight workers. Expected
wall time is approximately six to eight hours. The ceiling is a stop guard,
not a target; actual usage is reported.

## Scientific and claim boundary

The frozen confirmation remains:

- 24 entirely fresh catalytic matrices;
- both candidates, never pooled to rescue disagreement;
- natural landmarks 20, 40, and 60;
- PHI_UP, PHI_DOWN, RANDOM, and NOOP;
- 64 F8 futures per arm with fixed branch halves;
- common random streams across arms;
- complete exact replay and matrix-level inference.

Any positive result is only prospective confirmation of the pilot-developed
material full-block selector. It cannot rescue the public nine-atom Phi-r,
make the incomplete original 24-matrix development gate eligible, or establish
Phi as a physical cause, consciousness, agency, life, or a universal
origin-of-life mechanism.
