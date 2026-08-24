# Codex P3c catalytic-throughput clarification program

Status: implementation draft.  It becomes prospective only when the validation
bundle, external-response archive, protocols, seed registry, frozen predictor,
and source hashes are sealed into a registration before any P3c scientific
matrix is generated.

P3c is additive.  It does not modify or rescue P3b.  P3b remains a causal
`LOOSEN - TIGHTEN` result whose formal specificity gate failed because its
high-dose balanced-log random control was not equivalent to NOOP.

## Fixed intervention arms

At every eligible restored state, with present set `P`:

- `LOOSEN`: divide every `beta[P,P]` edge by 1.5;
- `TIGHTEN`: multiply every `beta[P,P]` edge by 1.5;
- `BALANCED_LOG_RANDOM`: the exact P3b within-block, zero-sum-log perturbation
  with Frobenius distance `0.5 * ||beta[P,P]||_F`;
- `THROUGHPUT_NEUTRAL_RANDOM`: a random additive perturbation of all and only
  `P x P`, with the same Frobenius distance, strict positivity, and exactly
  preserved starting `x.T @ beta @ x` within registered float64 tolerance;
- `NOOP`: unchanged beta.

For the neutral arm, let `b = vec(beta[P,P])` and
`w = vec(x[P] x[P].T)`.  Draw `z` from its sealed selection stream, form
`d0 = b * z`, project `d0` off `w`, normalize to the target norm, choose a
random first sign and then its opposite if needed, and accept only a strictly
positive block.  There are at most 4,096 attempts and no clipping.  Failure for
an otherwise eligible state aborts before its futures are launched.  States
with fewer than two present types are retained as all-arm structural no-ops.

## Frozen endpoint and simulator

The endpoint is JOINT_BREAK_RUN3 within F12: an inheritance break
`H <= 0.9`, followed strictly later by three consecutive inherited boundaries
`H > 0.9`.  Candidate 02 and candidate 03 retain their sealed Codex simulator,
fission, daughter, extinction, retry, and landmark contracts.  Future seeds do
not contain arm identity, so arms receive common random streams.

## Development pilot

- 40 fresh matrices shared across candidates;
- landmarks 20, 35, 50, 60, 65, 80;
- 32 F12 futures per arm/state, halves A=0-15 and B=16-31;
- 76,800 primary futures and a complete 76,800-future replay;
- primary advancement scope: landmarks 20, 35, 50, 65, 80;
- landmark 60: separately reported Fable-compatibility scope.

Before pilot outcomes are opened, the complete confirmation design below is
frozen.  The pilot advances only if all four candidate/half cells have:

1. positive `q_LOOSEN - q_TIGHTEN`;
2. negative state-centred slope of realised outcome shift on
   `log(T_arm / T_NOOP)`;
3. absolute `THROUGHPUT_NEUTRAL_RANDOM - NOOP <= 0.025`;
4. exact surgery audit, replay, and artifact readback.

These are advancement rules, not confirmation claims.  No pilot result may
change the confirmation design.

## Untouched confirmation

- 160 fresh matrices in a disjoint seed domain;
- otherwise the same six-landmark, five-arm, 32-branch design;
- 307,200 primary futures and a complete 307,200-future replay;
- whole catalytic matrix is the inference unit;
- 4,096 matrix bootstrap draws and 4,096 paired sign randomizations;
- Holm correction across the four candidate/half cells.

Every primary cell must pass all of:

1. `LOOSEN - TIGHTEN > 0`, its 95% bootstrap lower bound is positive, and
   Holm-adjusted randomization `p < 0.05`;
2. the state-centred slope on `log(T_arm/T_NOOP)` is negative and its 95%
   bootstrap upper bound is negative;
3. mean within-state Spearman association is negative and its 95% matrix
   bootstrap upper bound is negative;
4. the 90% matrix-bootstrap interval for
   `THROUGHPUT_NEUTRAL_RANDOM - NOOP` lies strictly inside `[-0.025, +0.025]`;
5. exact replay and artifact readback pass.

The balanced-log arm is diagnostic, not a required null.  Its outcome shift
will be reported against its throughput shift.

## Resistance

The confirmation futures also provide the prospectively registered resistance
analysis: first inheritance break within F6.  Report the same target contrast,
neutral-control equivalence, and throughput slope in every cell.  This does not
condition on a treatment-created break.

## Resilience from a shared natural break

Run only after the primary confirmation passes.  For every confirmation
matrix/candidate/standard landmark, use one fixed natural acquisition lineage
for at most 12 fissions.  If a natural break occurs, save the exact selected
daughter immediately after it and restore that identical broken state across
all five arms.  Do not retry, replace, or use an intervention to create the
break.  Matrices without any eligible broken state remain reported and
ineligible.  Require at least 120 eligible matrices per candidate or classify
resilience as inconclusive.

Launch 32 F8 futures per arm from each eligible shared state, in fixed halves,
plus complete replay.  In every candidate/half cell require:

- `run3_TIGHTEN - run3_LOOSEN > 0`, positive 95% lower bound, and
  Holm-adjusted paired-sign `p < 0.05`;
- positive state-centred run3 slope on `log(T_arm/T_NOOP)` with positive 95%
  lower bound;
- neutral random TOST equivalence to NOOP within +/-0.025;
- exact replay and readback.

Run5, time to renewal, inherited-boundary count, old-anchor similarity,
survival, entropy, and occupied types are secondary.

## Stops, budget, and claims

There is a mandatory stop after the pilot, confirmation, and resilience stage.
No feedback experiment is part of P3c.  No failed state or matrix is replaced;
intervention futures are never retried; candidates are never pooled to rescue
disagreement.  The intended CPU budget is about 14-20 CPU-hours and may not
exceed 30 CPU-hours without a new user decision.

A complete confirmation may support a Codex-specific causal catalytic-
throughput axis for hereditary break-and-renewal.  Cross-clean-room specificity
requires a compatible independently archived Fable geometry result.  P3c may
not support claims about life, agency, biological memory, real chemistry,
strict-eight control, autonomous attractors, or a universal origin-of-life
mechanism.
