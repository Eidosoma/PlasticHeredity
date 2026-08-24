# External P3c geometry request to Fable

This request must be answered and archived before the prospective P3c
registration is sealed.  The answer is external-hypothesis context only and
must not be used to tune Codex's arms, dose, endpoints, cohort size, gates, or
analysis.

Please report, on the states used for your beta-surgery experiment, the
per-state changes (arm minus no intervention, and ratios where appropriate) in:

1. `x.T @ beta @ x`;
2. the sum of the present-present beta block;
3. the Frobenius norm of that block;
4. the Perron spectral radius of that block;

for:

- `TIGHTEN = beta[P,P] * 1.5`;
- `LOOSEN = beta[P,P] / 1.5`;
- the historical random-surgery arm;
- if feasible, an exact within-`P x P`, balanced-log random perturbation with
  achieved Frobenius norm `0.5 * ||beta[P,P]||_F`.

Please also report whole-matrix uncertainty intervals for random-minus-NOOP
JOINT_BREAK_RUN3 probability, state whether these are descriptive or formal
equivalence intervals, and confirm both conventions:

- `beta[target, catalyst]` in the kinetics `beta @ n`;
- assembly catalytic throughput is `x.T @ beta @ x`.

Please distinguish values regenerated after this request from values that were
part of the originally sealed Fable result.
