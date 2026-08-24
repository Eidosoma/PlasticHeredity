# CR6 preregistration: zero-shot parameter-regime transfer

Status: frozen before any CR6 scientific matrix is generated.

## Question

CR1 prospectively confirmed that the immutable Codex 5x full-composite
predictor can select one-molecule edits that causally change F12
`JOINT_BREAK_RUN3` probability in the home beta regime `(A, sigma) = (-4, 4)`.

CR6 asks whether that exact predictor and edit-selection algorithm transfer
without retraining to three altered catalytic-matrix distributions, while
correctly predicting little or no control in a fourth weak-heterogeneity
distribution.

CR1 and all later results remain unchanged. CR5R was not used to select the
CR6 regimes, model, arms, sample sizes, endpoints, margins, or gates.

## Frozen simulator, endpoint, model, and edits

The sealed candidate 02 and 03 simulator contracts are unchanged. The endpoint
is `JOINT_BREAK_RUN3` within F12: a strict inheritance break (`H <= 0.9`)
followed strictly later by three consecutive inherited fissions (`H > 0.9`).
Unrounded float64 values are used. Certification before later extinction stays
positive; extinction before certification is negative.

The model is the exact candidate-separated 5x full-composite predictor copied
from the sealed CR1 registration. Its feature map, scalers, PCA transforms,
coefficients, priors, and prediction mapping are immutable. CR6 performs no
refitting, recalibration, threshold adjustment, regularization search,
candidate-specific rescue, or regime-specific family switching.

At each restored post-fission state, every legal mass-preserving one-molecule
substitution is scored exhaustively. `MODEL_UP` is the first deterministic
maximum, `MODEL_DOWN` the first deterministic minimum, `RANDOM` one uniformly
sampled legal substitution from a separate stream, and `NOOP` the unchanged
state. Already observed history is held fixed during the instantaneous edit.

## Registered regimes and cohort

The four beta distributions are:

| Key | A | sigma | Registered role |
|---|---:|---:|---|
| `POS_A_M4_S5` | -4 | 5 | positive transfer |
| `POS_A_M3_S4` | -3 | 4 | positive transfer |
| `POS_A_M5_S4` | -5 | 4 | positive transfer |
| `NULL_A_M4_S3` | -4 | 3 | predicted null |

For each regime:

- 40 completely fresh catalytic matrices shared across candidates;
- both Codex candidates;
- untreated natural landmarks 35 and 65;
- 160 restored states;
- arms `MODEL_UP`, `MODEL_DOWN`, `RANDOM`, and `NOOP`;
- 48 F12 futures per arm and state;
- fixed halves A = branches 0--23 and B = branches 24--47;
- complete exact replay of all 30,720 futures.

Across four regimes this is 640 restored states, 122,880 primary futures, and
122,880 replay futures. No scientific matrix, state, or branch is retried or
replaced. The optional physical-rule arms are omitted from CR6; they would be a
different registered intervention family.

## Randomness

Every regime and purpose has a distinct sealed seed domain. The matrix,
main-trajectory, random-edit-selection, future, bootstrap, randomization, and
replay domains are separated. Within a regime, candidate, matrix, landmark,
and branch, all arms receive the same arm-free future seed. These are common
random streams, not necessarily identical realized futures after edits cause
the trajectories to diverge. Random edit selection never consumes the future
stream.

## Positive-transfer inference and gates

The catalytic matrix is the inference unit. Both landmarks, arms, and branches
from a matrix travel together. Each regime independently uses 4,096
whole-matrix bootstrap draws and 4,096 paired whole-matrix sign
randomizations. Holm adjustment is across its four candidate-by-branch-half
cells.

Each of the first three regimes passes only if all four cells have:

1. `MODEL_UP - MODEL_DOWN > 0`;
2. a 95% whole-matrix bootstrap lower bound above zero;
3. Holm-adjusted one-sided matrix-randomization `p < 0.05`; and
4. `RANDOM` equivalent to `NOOP` by a TOST margin of `+/-0.025`, implemented
   as a 90% whole-matrix bootstrap interval strictly inside the margin.

`MODEL_UP - NOOP`, `NOOP - MODEL_DOWN`, and the random-to-targeted effect ratio
are reported but are not CR6 transfer gates. One regime, candidate, or half
cannot rescue another.

## Predicted-null inference and gate

For `NULL_A_M4_S3`, each candidate is evaluated using all 48 branches. State
effects are first averaged within catalytic matrix, then bootstrapped by whole
matrix. The targeted effect is classified as equivalent to zero only if its
90% bootstrap TOST interval lies strictly inside `+/-0.04` in both candidates.
A confidence interval merely crossing zero is not equivalence. Candidate-half
results and random-minus-no-op are reported descriptively but are not allowed
to replace the candidate-pooled null gate.

The complete CR6 prediction passes only if all three positive-transfer regime
gates, both candidate null-equivalence gates, exact replay, written-artifact
readback, and checksums pass.

## Resources and stop

CR6 requires at least 3 GB free at launch and accepts a declared CPU budget of
3--6 hours. Expected use is approximately 3--5 CPU-hours and 15--30 minutes
wall time with 14 workers. This keeps the recent CR5 + CR5R + CR6 sequence
below the requested 30 CPU-hour ceiling.

The campaign is checkpoint-resumable and stops after the CR6 seal. CR7 is not
launched automatically.

## Claim boundary

If all three positive regimes pass, Codex may claim zero-shot transfer of its
frozen molecular control law across those registered beta distributions. If
the fourth regime passes its TOST gate, Codex may additionally claim that the
registered predicted weak-control regime was confirmed as equivalent to zero
within `+/-0.04`.

CR6 cannot establish universal transfer, control of strict-eight, biological
memory, agency, life, autonomy, an installed attractor, real prebiotic
chemistry, a universal origin-of-life mechanism, or Phi/PhiID intervention.
