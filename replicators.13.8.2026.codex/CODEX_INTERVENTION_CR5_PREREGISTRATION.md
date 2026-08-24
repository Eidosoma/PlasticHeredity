# CR5 preregistration: resistance and resilience

Status: design frozen before CR5 development-label generation and before any
CR5 confirmation matrix is generated.

## Question

The already confirmed `JOINT_BREAK_RUN3` endpoint combines two processes:

1. resistance: whether heredity breaks at all; and
2. resilience: whether three consecutive inherited fissions resume after an
   already observed natural break.

CR5 tests these processes separately with mass-preserving one-molecule edits.
It does not target the strict-eight endpoint.

## Simulator and thresholds

The sealed Codex candidate 02 and 03 simulator contracts are unchanged.
Inheritance is the strict unrounded float64 comparison `H > 0.9`.
Extinction before an endpoint is certified is negative; an endpoint certified
before later extinction remains positive. Intervention futures are never
retried and failed matrices or lineage sources are never replaced.

## Development and model freeze

The 200-matrix, two-candidate, five-landmark 5x development cohort (`VALI`) is
reconstructed from its archived seed and must reproduce its archived feature
arrays and F12 `JOINT_BREAK_RUN3` labels exactly under a one-thread numerical
guard. Those matrices have never contributed to an intervention confirmation.

The break target is first break within F6, derived from exact replay of the
archived F12 development futures (32 branches per state). For resilience, each
untreated development landmark is advanced under a separate registered stream
for at most 60 fissions until its first natural break. The exact selected
daughter immediately after that break is saved. Sources that become extinct or
show no break are retained in the acquisition ledger and are not replaced.
Eligible broken daughters receive 32 independent F8 development futures; the
target is any run of three consecutive strict inheritances within those eight
fissions.

Two candidate-separated students are fitted for each candidate:

- `q_B(s)`: probability of a first break within F6;
- `q_R(s)`: probability of run3 within F8 from an already broken daughter.

Both use one architecture: standardize the existing 195 state/graph features,
retain 12 full-SVD PCA components, append the nine existing past-observable
history features, standardize the 21-coordinate combined vector, and fit a
binomial logistic model. The intercept is unpenalized and all 21 coefficients
receive the same L2 penalty. Candidate-specific penalties are selected by
five-fold whole-matrix cross-validation from exactly
`{0.001, 0.01, 0.1, 1, 10, 100}`. Matrix fold is `matrix_id mod 5`; ties within
`1e-12` choose the larger penalty. All fold transforms are fitted using only
their training matrices. The development futures are replayed completely, and
the resulting transforms, coefficients, penalties, prediction mapping, and
hashes are frozen before confirmation generation.

## Stage A: resistance

- 200 completely fresh catalytic matrices shared across candidates;
- both candidates;
- natural landmarks 20, 35, 50, 65, and 80;
- arms `BREAK_UP`, `BREAK_DOWN`, `RANDOM`, and `NOOP`;
- exhaustive scoring of every legal mass-preserving substitution using `q_B`;
- 64 F6 futures per arm and state;
- fixed halves A = 0--31 and B = 32--63;
- complete exact replay.

Primary target: at least one strict break within F6.

## Stage B: resilience

Each of the same untreated confirmation landmark sources is advanced under a
new, arm-free acquisition stream for at most 60 fissions until its first
natural break. The exact post-break daughter is restored across all arms before
any edit. Arms are `RENEWAL_UP`, `RENEWAL_DOWN`, `RANDOM`, and `NOOP`, selected
exhaustively with `q_R`. Each eligible broken state receives 64 F8 futures per
arm, in the same fixed halves, followed by complete exact replay.

The primary target is run3 within F8. Secondary outcomes are run5, time to
run3, inherited-boundary count, similarity to the pre-break parent, survival,
growth updates, final entropy, and occupied types. If either candidate lacks
at least one eligible broken state from every one of the 200 matrices, Stage B
is sealed inconclusive without replacing matrices.

## Randomness

All seed domains are purpose-keyed and sealed. Random-edit selection never uses
the future stream. For a phase, candidate, matrix, landmark, and branch, every
arm receives a generator initialized from the same arm-free seed. These are
common random streams, not necessarily identical realized futures after edited
states diverge.

## Inference and gates

The catalytic matrix is the inference unit. All landmarks and eligible broken
states belonging to one matrix remain together. Each stage uses 4,096
whole-matrix bootstrap draws and 4,096 paired whole-matrix sign randomizations.
Holm adjustment is across the four candidate-by-half cells within that stage.

A stage passes only when every cell has:

1. targeted up-minus-down effect greater than zero;
2. 95% whole-matrix bootstrap lower bound greater than zero;
3. Holm-adjusted randomization `p < 0.05`;
4. `RANDOM` equivalent to `NOOP` by a TOST margin of `+/-0.025`, implemented
   as a 90% bootstrap interval strictly inside the margin; and
5. absolute random-minus-noop no greater than 25% of up-minus-down.

Up-minus-noop and noop-minus-down are reported but are not additional CR5
gates. Candidate or half failures cannot be rescued by pooling.

## Integrity, budget, and stop

Validation, protocol registration, development replay/model freeze, a second
confirmation seal, and a non-scientific smoke test must all precede the first
confirmation matrix. The full campaign is budgeted at no more than 30 CPU
hours, requires at least 4 GB free at launch, and is checkpoint-resumable. No
mid-phase kill is planned. The run stops after sealed CR5 reporting; CR6 is not
launched automatically.

## Claim boundary

A passing resistance stage supports causal control of whether short-horizon
heredity first breaks. A passing shared-state resilience stage supports causal
control of short-run recovery after an identical naturally broken state. CR5
does not establish biological repair, memory, agency, life, autonomous
organization, strict-eight control, a universal origin-of-life mechanism, real
prebiotic chemistry, or Phi/PhiID intervention.
