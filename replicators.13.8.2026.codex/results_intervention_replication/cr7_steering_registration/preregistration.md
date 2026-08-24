# CR7 preregistration: closed-loop hereditary steering

Status: implementation specification to be checksum-sealed before any CR7
scientific matrix is generated.

## Scientific question and phase boundary

CR7 asks whether repeated, state-dependent one-molecule substitutions can
externally maintain different levels of hereditary stability for 60 fissions
in the two independently reconstructed Codex GARD candidates. It targets only
the validated `JOINT_BREAK_RUN3` process and its constituent inheritance
boundaries. It does not target strict-eight.

CR7 is authorized independently by the sealed CR1 model-guided and CR3
outgoing-rule confirmations. The failed complete CR6 gate is preserved and is
not used to choose, tune, or recalibrate any CR7 controller.

## Frozen inputs and fresh cohort

The candidate-separated 5x-development predictor is copied from the sealed
CR1 registration and must retain SHA-256
`9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af`.
There is no refit, recalibration, threshold change, family switch, or weight
fit. The physical controller is the sealed CR3 outgoing catalytic-influence
rule, `x @ beta == beta.T @ x`, under Codex storage
`beta[target, catalyst]`.

The scientific cohort contains 48 completely fresh catalytic matrices shared
across candidates. For each matrix and candidate, an untreated natural main
trajectory is generated using the established 100-attempt retry contract. The
single launch state is its restored generation-60 post-fission state. No
controlled lineage is retried or replaced.

Each controller receives six replicate lineages and each lineage is followed
for 60 fissions, for 3,456 primary controlled lineages and at most 207,360
primary fission boundaries. Every primary lineage is executed again in a
complete exact replay.

## Controllers

After every successfully completed fission, including the sixtieth boundary,
the active controller is called on the selected daughter state. The last edit
therefore affects the registered final-state summaries and action counts but
cannot affect an already observed boundary.

The six frozen controllers are:

1. `MODEL_UP`: exhaustively score every legal mass-preserving one-molecule
   substitution and apply the first lexicographic edit at the maximum frozen
   `JOINT_BREAK_RUN3` prediction.
2. `MODEL_DOWN`: use the corresponding minimum.
3. `RULE_UP`: remove the present type with the greatest outgoing influence
   and add the type with the least outgoing influence, implemented as the
   legal edit minimizing the change in `x @ beta` support score.
4. `RULE_DOWN`: reverse that direction, using the legal edit maximizing the
   support-score change.
5. `RANDOM`: uniformly sample one legal edit using a controller-action stream
   separate from simulation.
6. `NOOP`: invoke a callback that returns no edit.

The no-op callback is compared bitwise with the plain simulator for every
matrix, candidate, and replicate. A failed or extinct controlled lineage is
retained as observed and receives no retry.

## Randomness

For one phase, candidate, matrix, and replicate, arm identity is absent from
the future-simulation seed. These are common random streams, not identical
realized futures after edited paths diverge. Matrix generation, main
trajectory, random controller action, future simulation, bootstrap,
randomization, replay, and conditional extension have distinct purpose-keyed
domains. Random action selection cannot consume simulation randomness.

The catalytic matrix is the inference unit. All controllers and all six
replicate lineages from one matrix remain together in every resample.
Candidates are never pooled to rescue disagreement.

## Outcomes and exact definitions

Primary outcomes over the observed portion of each 60-fission lineage are:

- inherited-boundary fraction, with strict inheritance `H > 0.9` on
  unrounded float64 values;
- total breaks (`H <= 0.9`);
- non-overlapping certified `JOINT_BREAK_RUN3` episode count; after each
  certification the state machine requires a new break before another episode
  can count, and a break during a pending renewal resets the trailing run;
- longest consecutive inherited run.

Registered secondary outcomes are final entropy, occupied molecular types,
top-1 abundance share, `x.T @ beta @ x` throughput, mean growth updates per
observed fission, survival through 60 fissions, mean pairwise final-composition
cosine similarity across the six same-matrix replicates, distinct swaps,
repeated swaps, immediately reversing swaps, frozen risk before and after
actions, and out-of-development-envelope fraction.

The development envelope is frozen from every transformed 21-coordinate
full-model input in the original 5x development cohort, separately by
candidate. A decision state is out of envelope if any post-action transformed
coordinate lies strictly below its candidate-specific development minimum or
strictly above its maximum. This is descriptive and never changes an action.

## Inference and gates

The same 4,096 whole-matrix bootstrap index draws are used for every
candidate, arm, outcome, and contrast. Descriptive paired whole-matrix sign
randomizations use 4,096 draws; CR7's registered decisions are the confidence
interval gates below. Random-versus-no-op equivalence uses a 90% whole-matrix
bootstrap interval and the already established `+/-0.025` inheritance-fraction
margin. A confidence interval crossing zero is not equivalence.

CR7 passes only if exact replay and the full no-op audit pass and, separately
in both candidates:

1. `MODEL_DOWN - NOOP` inherited fraction is positive with 95% lower bound
   above zero;
2. `RULE_DOWN - NOOP` inherited fraction is positive with 95% lower bound
   above zero;
3. `MODEL_UP - NOOP` inherited fraction is negative with 95% upper bound
   below zero;
4. `MODEL_UP - MODEL_DOWN` episode count is positive with 95% lower bound
   above zero; and
5. `RANDOM - NOOP` inherited fraction is TOST-equivalent within `+/-0.025`.

The physical-rule recovery fraction is
`(RULE_DOWN - NOOP) / (MODEL_DOWN - NOOP)`. A strong external-replication
classification additionally requires a point estimate of at least 0.80 and a
whole-matrix bootstrap lower bound above 0.70 in both candidates. This ratio
does not rescue a failed primary steering gate. Ratio bootstrap draws with a
nonpositive model-guided denominator are invalid; the strong classification
also requires at least 95% valid ratio draws.

## Conditional active-control extension

Only if every 60-fission stabilization gate passes in both candidates,
`MODEL_DOWN`, `RULE_DOWN`, and `NOOP` continue from their exact fission-60
states and saved simulation-generator states for another 60 fissions. The
extension protocol and seeds are frozen now; it is not selected using an
extension outcome. It is continued active feedback, not passive persistence.
The extension also receives a complete exact replay.

The phase seals and stops after reporting. CR8 and CR9 do not launch
automatically.

## Claim boundary

A passing CR7 may support externally maintained hereditary modes in these two
Codex simulator contracts while feedback is active. It cannot establish
autonomous persistence, an installed attractor, biological memory, agency,
life, error correction, real prebiotic chemistry, a universal origin-of-life
mechanism, Phi/PhiID control, or control of strict-eight. Release and return
must be tested separately before any autonomous claim.
