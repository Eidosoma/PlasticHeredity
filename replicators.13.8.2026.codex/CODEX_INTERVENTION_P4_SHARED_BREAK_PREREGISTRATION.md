# Codex P4 shared-natural-break recovery preregistration

P4 is a new additive experiment motivated by the sealed P3c result. It does
not continue P3c under its failed gate, does not rescue P3c, and is not the
molecular-student CR5 experiment in the full Fable directive.

## Scientific questions

Starting from the exact same naturally broken selected daughter:

1. Does coherent strengthening rather than weakening of the occupied
   catalytic web increase the probability of three consecutive inherited
   fissions within the next eight fissions?
2. Does changing occupied-network topology while preserving the daughter's
   starting `x^T beta x` have a reproducible effect, a negligible effect, or an
   unresolved effect on recovery?

The first question and the topology question are separate inferential
families. Topology is a treatment, not a required null.

## Cohort and acquisition

- 160 entirely fresh catalytic matrices shared across Codex candidates 02 and
  03;
- untreated restored landmarks 20, 35, 50, 65, and 80;
- one purpose-keyed untreated acquisition lineage per source state for at most
  12 fissions;
- the first naturally non-inherited selected daughter (`H <= 0.9`) is saved;
- every arm restores that identical daughter and its identical observed
  history;
- no intervention may create the qualifying break;
- no retry, matrix replacement, or adverse-lineage exclusion; and
- at least 120 eligible catalytic matrices per candidate, otherwise the phase
  seals as inconclusive before intervention futures are launched.

All source states belonging to one catalytic matrix remain together in every
bootstrap and randomization draw.

## Frozen arms

The occupied set is `P = {i: x_i > 0}` in the broken daughter.

- `TIGHTEN`: multiply `beta[P,P]` by 1.5.
- `LOOSEN`: divide `beta[P,P]` by 1.5.
- `THROUGHPUT_NEUTRAL_TOPOLOGY`: apply a positive present-present
  perturbation with exact Frobenius norm `0.5 ||beta[P,P]||_F`, exactly
  orthogonal to the daughter's throughput weights, so launch-state
  `x^T beta x` is unchanged.
- `BALANCED_LOG_RANDOM`: the archived P3c balanced-log diagnostic; it is not a
  required null.
- `NOOP`: unchanged beta.

Singleton occupied sets remain in the cohort and use structural no-op for all
arms.

## Futures and endpoint

- 32 F8 futures per arm and eligible state;
- branch half A is 0--15 and half B is 16--31;
- arm identity is absent from the future seed;
- all arms receive common random streams, not guaranteed identical realised
  random draws after paths diverge;
- complete deterministic acquisition and future replay.

The primary recovery endpoint is three consecutive strict inherited
boundaries (`H > 0.9`) within F8. No additional break is required because the
launch state is already the exact daughter following a natural break.

Secondary outcomes are run5, time to run3, inherited-boundary count, survival,
similarity to the pre-break parent, entropy, occupied types, and growth
updates.

## Frozen inference and classifications

Use 4,096 whole-matrix bootstrap draws and 4,096 paired whole-matrix sign
randomizations.

### Catalytic-strength recovery

In candidate 02 half A, candidate 02 half B, candidate 03 half A, and candidate
03 half B, require:

- `q_TIGHTEN - q_LOOSEN > 0`;
- positive 95% matrix-bootstrap lower bound;
- Holm-adjusted one-sided sign-randomization `p < 0.05` across the four cells;
- positive state-centred slope of realised recovery shift on
  `log(T_arm/T_NOOP)`; and
- positive 95% bootstrap lower bound for that slope.

All four cells and exact replay must pass for a confirmed strength effect on
recovery.

### Fixed-throughput topology

For `THROUGHPUT_NEUTRAL_TOPOLOGY - NOOP`, use two-sided inference and classify:

- **reproducible directional topology effect** only if all four cells have the
  same nonzero sign, every 95% interval excludes zero, and every Holm-adjusted
  two-sided sign-randomization p-value is below 0.05;
- **negligible within +/-0.025** only if every 90% interval lies strictly
  inside that equivalence region; or
- **inconclusive** otherwise.

The topology classification cannot invalidate or rescue the separate
catalytic-strength recovery result.

## Stops and boundaries

P4 stops after its sealed result. It does not launch feedback, molecular
students, or any other Fable phase. It may support causal recovery from an
identical broken state in Codex's simulations. It cannot establish biological
repair, memory, agency, life, an autonomous attractor, real chemistry, or a
universal origin-of-life mechanism.

