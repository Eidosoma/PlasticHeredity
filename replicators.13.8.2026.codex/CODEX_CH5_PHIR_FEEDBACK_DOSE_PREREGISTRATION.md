# Codex Chapter 5 D24: feedback-strength dose reconciliation

Status: prospective registration document. Scientific results may be written
only after this document, the implementation, tests, frozen predictor, seed
domains, and analysis contract have been validated and source-hash sealed.

## Motivation and separation

The completed Codex Chapter 5 pilot and fresh window bridge remain unchanged.
The bridge established that hereditary stabilization increased inheritance by
about 0.23--0.28 while the revised nine-atom Phi-r contrast was negative under
both pooled and rolling temporal estimators. A completed external clean-room
remeasurement reported a smaller hereditary contrast, about 0.11--0.16, and a
positive revised Phi-r contrast under both estimators. Window construction
therefore does not explain the sign disagreement.

This new D24 campaign prospectively tests one narrower hypothesis: the revised
Phi-r response may be nonlinear in the strength of repeated predictor-guided
molecular control. It does not repeat, amend, pool with, rescue, or overwrite
either completed Codex result, and it cannot authorize the sealed 48-matrix
Chapter 5 confirmation.

No external code, matrices, states, seeds, models, trajectories, selected
edits, or result objects enter D24. External values are descriptive comparison
benchmarks only.

## Frozen cohort and simulator contracts

- 24 entirely fresh catalytic matrices from new purpose-separated seeds.
- Codex candidates 02 and 03, analyzed separately.
- Two natural replicates per candidate and matrix, analyzed separately.
- Natural launch after 60 untreated fissions under the existing bounded
  natural-path retry contract.
- The immutable Codex 5x JOINT_BREAK_RUN3 predictor with expected SHA-256
  `9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af`.
- Eleven 60-fission feedback arms sharing common future random streams within
  candidate, matrix, and replicate. Arm identity is absent from future keys.
- Complete deterministic regeneration into a second checkpoint tree.
- No failed, extinct, or adverse scientific matrix is replaced.

The instantaneous molecular edit holds observed history fixed. Subsequent
history evolves normally. Intervention futures are never retried.

## Frozen feedback-strength selector

At every post-fission daughter state, enumerate and exactly score every legal
mass-preserving one-molecule substitution. Let `p0` be the no-op prediction.
Choose the legal `NEUTRAL` edit whose prediction is closest to `p0`, breaking
ties by `(remove_type, add_type)`. Let `p_min` and `p_max` be the deterministic
minimum- and maximum-risk legal edits.

For alpha in `{0.25, 0.50, 0.75, 1.00}`, define:

    stabilizing_target = p_neutral + alpha * (p_min - p_neutral)
    destabilizing_target = p_neutral + alpha * (p_max - p_neutral)

Select the legal edit closest to the target, again breaking ties by molecular
indices. At alpha 1 the selector must reproduce the frozen exhaustive
minimum/maximum selector exactly.

The eleven arms are:

- `NOOP`;
- `RANDOM`, one uniformly selected legal edit from a separate action stream;
- `NEUTRAL`;
- `STABILIZE_25`, `STABILIZE_50`, `STABILIZE_75`, `STABILIZE_100`;
- `DESTABILIZE_25`, `DESTABILIZE_50`, `DESTABILIZE_75`, `DESTABILIZE_100`.

Every edited arm applies exactly one edit after every completed fission.
Persist the no-op, neutral, extreme, target, and selected predictions; selected
edit; requested alpha; achieved local alpha; and deterministic action digest.

## Observation and endpoints

The molecular stream begins at the restored launch state and then contains,
in order, every growth update, selected daughter, and (for edited arms) the
post-edit composition. Fission and intervention transitions are unmasked.

Primary outcomes over controlled fissions 31--60:

1. strict inherited-boundary fraction (`H > 0.9`);
2. one pooled-final-30 CLR/drop-last revised nine-atom Phi-r reading;
3. all 16 corresponding PhiID atoms and the registered causation, emergence,
   and synergy-persistence summaries.

Registered secondary readings are the mean rolling-512 revised Phi-r over
fissions 31--60; pooled full-dimensional and macro typeset readings; the
normalized-full ratio; pooled raw-count revised sensitivity; and RANDOM/NOOP
and NEUTRAL/NOOP contrasts. Raw trajectories are never persisted.

## Estimands, inference, and classification

For each nonzero alpha, the dose contrast is stabilizing minus destabilizing
within matrix, candidate, and replicate. The catalytic matrix is the inference
unit. All arms and repeated readings from one matrix remain together in every
draw. Use 4,096 whole-matrix bootstrap draws and 4,096 paired whole-matrix sign
randomizations. Apply Holm correction across the four candidate-by-replicate
cells within every registered family. Candidates and replicates are never
pooled to rescue disagreement.

The controller-strength dial is valid only if, in all four cells:

- the alpha-1 hereditary contrast is positive with positive 95% lower bound
  and Holm-adjusted upper-tail `p < 0.05`; and
- the within-matrix linear slope of hereditary contrast against alpha is
  positive with positive 95% lower bound and adjusted `p < 0.05`.

The high-dose Codex response is reproduced only if the alpha-1 pooled revised
Phi-r contrast is negative with negative 95% upper bound and Holm-adjusted
lower-tail `p < 0.05` in all four cells.

The strict strength-explanation gate passes only if the dial is valid and all
four cells satisfy:

1. alpha-0.5 pooled revised Phi-r is positive with positive lower bound and
   adjusted upper-tail `p < 0.05`;
2. alpha-1 pooled revised Phi-r is negative with negative upper bound and
   adjusted lower-tail `p < 0.05`;
3. the paired matrix-level alpha-0.5 minus alpha-1 Phi-r contrast is positive
   with positive lower bound and adjusted upper-tail `p < 0.05`.

If only condition 3 and dial validity pass in all cells, classify the result as
partial dose moderation. Otherwise classify it as no registered evidence that
control strength explains the sign disagreement. Results at alpha 0.25 or
0.75 are descriptive and cannot rescue the strict gate. The achieved alpha-0.5
hereditary contrast is compared descriptively with the external 0.11--0.16
range and is not a fitting or pass target.

## Validation, execution, and stop rule

Before a scientific matrix exists, validation must cover edit legality and
mass preservation, exhaustive enumeration, neutral and interpolated selection,
deterministic ties, monotone target selection, alpha-1 equality with the frozen
extrema, common future streams, separate random-action streams, no-op simulator
parity, unchanged instantaneous history, score/atom identities, matrix-block
inference, checkpoint restart, serialization, source/model hashes, complete
replay, and preservation of the locked confirmation.

A non-scientific smoke run may exercise every path but must not disclose arm
ordering, scientific rates, effect sizes, or candidate differences. The 24
matrix campaign must run detached with durable per-matrix checkpoints. It
stops after replay, analysis, sealing, and user review. It never launches a
48-matrix continuation.

## Claim boundary

A passing strict gate would support only that intervention strength explains
the sign change across the prospectively specified Codex dose levels. Partial
moderation would support sensitivity to dose without explaining the external
sign disagreement. Neither result selects a uniquely correct Phi-r, makes
Phi-r a controller or cause, or supports consciousness, agency, life,
biological memory, a universal origin-of-life mechanism, real prebiotic
chemistry, or a literal Platonic-space/Ruliad interpretation.
