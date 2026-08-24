# Codex P3b beta-surgery dose-and-contract bridge

Status: this protocol and its executable source must be checksum-sealed before
the first P3b scientific matrix is generated.

## Why P3b exists

The sealed Codex P3 result is preserved unchanged.  It implemented the written
instruction `0.05 * ||beta[P,P]||_F` exactly and found a mixed small-dose result:
no combined effect in candidate 02 and a small same-direction effect in
candidate 03.  After that result was sealed, Fable confirmed that the `0.05`
instruction was an assembly error.  Its actual frozen intervention used
`delta=0.5`:

- raise/tighten: `beta[P,P] *= 1.5`;
- lower/loosen: `beta[P,P] /= 1.5`.

The Fable pair is symmetric in log space (`+/- log(1.5)`) but asymmetric in
Frobenius distance: tightening changes the block by `0.5 * ||beta[P,P]||_F`,
while loosening changes it by `(1/3) * ||beta[P,P]||_F`.  This was intentional.

P3 is therefore a valid, unintended small-dose experiment and is not a failed
replication of Fable's registered surgery.  P3b prospectively tests the actual
Fable-strength contract and repeats the small dose on the same new cohort to
provide two-dose evidence.

## Fixed simulator and endpoint

P3b uses the unchanged Codex candidates, simulator, beta convention
`beta[target,catalyst]`, and `JOINT_BREAK_RUN3` endpoint.  It does not import or
use Fable code, models, matrices, states, seeds, branches, or result files.

The endpoint remains a break (`H <= 0.9`) followed strictly later by three
consecutive inherited fissions (`H > 0.9`) within F12.  A certified event before
later extinction remains positive; extinction before certification is negative.

## Cohort and computation

- 80 entirely fresh catalytic matrices, shared across candidates.
- Candidates 02 and 03 analyzed separately.
- Natural post-fission landmarks 20, 35, 50, 60, 65, and 80.
- 960 restored states.
- 32 F12 futures per arm and state.
- Fixed branch halves A=0--15 and B=16--31.
- Seven arms and 215,040 futures per pass.
- Complete deterministic replay of all 215,040 futures.
- No state selection by risk, history, propensity, or future outcome.
- No intervention-future retry and no matrix replacement.

## Frozen arms

For `P = {i: x_i > 0}`, every non-noop surgery changes only `P x P` and remains
active for the complete F12 future.  Composition and observed history are
identical at launch.

1. `SMALL_LOOSEN`: multiply `beta[P,P]` by `0.95`.
2. `SMALL_TIGHTEN`: multiply `beta[P,P]` by `1.05`.
3. `SMALL_RANDOM_PP`: change every `P x P` edge along one zero-mean log-space
   random direction, scaled numerically to exact Frobenius norm
   `0.05 * ||beta[P,P]||_F`.
4. `FABLE_LOOSEN`: divide `beta[P,P]` by `1.5`.
5. `FABLE_TIGHTEN`: multiply `beta[P,P]` by `1.5`.
6. `FABLE_RANDOM_PP`: use the same state-specific random log direction as the
   small random arm, scaled to exact Frobenius norm
   `0.5 * ||beta[P,P]||_F`.
7. `NOOP`: unchanged beta.

The random controls are stricter than Fable's historical whole-matrix random
arm.  They control location, number of changed edges, positivity, and achieved
norm: all and only present-present edges change; logarithmic directions sum to
zero; positivity is intrinsic; and the achieved post-change norm is audited.
They do not reproduce Fable's acknowledged duplicate-edge, clipping, and norm
shortfall behavior.

Random-surgery direction streams are separate from all future streams.  Future
seed keys omit arm identity, so paired arms receive common random streams, not
necessarily identical realized futures after divergence.

## Primary replication analysis

The primary compatibility cohort is landmark 60, matching Fable's original
surgery state.  Evaluate separately in candidate 02 half A, candidate 02 half
B, candidate 03 half A, and candidate 03 half B.

For each cell define:

    D_F = q(FABLE_LOOSEN) - q(FABLE_TIGHTEN)

The primary gate passes only if all four cells have:

1. `D_F > 0`;
2. a positive 95% whole-matrix bootstrap lower bound;
3. Holm-adjusted whole-matrix sign-randomization `p < 0.05`;
4. `FABLE_RANDOM_PP` equivalent to `NOOP` by a 90% whole-matrix bootstrap
   TOST interval wholly inside `[-0.025, +0.025]`; and
5. exact replay.

There is no added requirement that each targeted arm separately differ from
no-op, and no random-effect-ratio gate.  Those are CR1 conditions and are not
part of this CR4 bridge.

## Registered secondary analyses

### Five-landmark generalization

Repeat the primary high-dose analysis after matrix-level averaging over
landmarks 20, 35, 50, 65, and 80.  Use its own four-cell Holm family.  Failure
cannot be rescued by the landmark-60 result, and vice versa.

### Two-dose ordering

Within both the landmark-60 and five-landmark scopes define:

    D_S = q(SMALL_LOOSEN) - q(SMALL_TIGHTEN)
    D_D = D_F - D_S

Report `D_S` with intervals.  A registered graded two-dose result requires all
four cells to have `D_D > 0`, a positive 95% bootstrap lower bound, and a
Holm-adjusted sign-randomization `p < 0.05`.  Also report formal equivalence of
`SMALL_RANDOM_PP` to `NOOP`.  This supports two-dose ordering, not a complete
dose-response curve.

### Other outcomes

Report break within F12, run3 after break, inherited-boundary count, first-break
time, renewal-certification time, survival, growth updates, entropy, and
occupied types.  Report break hazard separately; do not attribute an effect
specifically to renewal without a shared-broken-state experiment.

## Inference and audit

- Catalytic matrix is the inference unit.
- All states, landmarks, arms, and halves from a matrix stay together.
- 4,096 shared whole-matrix bootstrap draws.
- 4,096 shared paired whole-matrix sign randomizations.
- Complete edge tables and per-state norm audits are persisted.
- All predictions are descriptive; no model selects a surgery.
- No refitting, recalibration, threshold change, or outcome-dependent choice.
- Complete written-artifact readback must reproduce inference exactly.
- Complete deterministic replay is mandatory.

## Stop and claim boundary

The P3b result is sealed and reported whether positive, mixed, or negative.
No later intervention phase launches automatically.

A passing landmark-60 primary gate may support qualitative cross-clean-room
replication that Fable-strength present-present catalytic surgery causally
changes `JOINT_BREAK_RUN3` under the independent Codex simulator contracts.  A
passing five-landmark gate adds Codex-landmark generalization.  A passing
two-dose gate supports graded two-dose ordering.

It cannot establish strict-eight control, Phi/PhiID intervention, biological
memory, autonomy, agency, life, error correction, real prebiotic chemistry, or
a universal origin-of-life mechanism.
