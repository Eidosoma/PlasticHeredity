# Codex intervention CR9 preregistration

## Scientific question

CR7 established strong hereditary stabilization while state-dependent molecular
feedback remained active. CR8 found no registered nonzero restoring basin after
release. CR9 asks two bounded follow-up questions:

1. Does a longer period of active steering leave a longer-lived transient after
   steering stops?
2. How infrequently or selectively can the frozen corrective action be applied
   while still improving inheritance over untreated and budget-matched random
   intervention?

Transient persistence is not an autonomous restoring basin. Sparse active
feedback is not passive memory.

## Frozen upstream contracts

CR9 uses the unchanged Codex candidate-02 and candidate-03 simulator contracts,
the immutable 5x-development `JOINT_BREAK_RUN3` predictor used by CR1 and CR7,
and the exact CR7 `MODEL_DOWN` edit rule: exhaustively score every legal
mass-preserving one-molecule substitution and take the lowest predicted-risk edit,
with first-lexicographic tie resolution. No refitting, recalibration, threshold
search, candidate-specific rescue, or Fable object is permitted.

CR7's sealed sixty-fission stabilization gate is the phase prerequisite. CR8 is
recorded as context but neither authorizes nor tunes CR9.

## Fresh cohort

Before generating any CR9 matrix, freeze 48 new catalytic-matrix seeds, a new
initial-composition domain, and a new untreated main-trajectory domain. Each
matrix is shared across both candidates. The launch state is the natural,
untreated post-fission landmark at generation 60. The already frozen main-path
retry contract is allowed only while creating that landmark. No policy lineage is
retried or replaced.

Every experiment uses six replicate lineages per candidate/matrix/policy. The
catalytic matrix is the inference unit. Replicates, policies, time points, and
candidates' within-matrix material stay in their registered blocks, while
candidate conclusions remain separate.

## CR9-A: steering-pulse ladder

Starting from each natural generation-60 state, apply `MODEL_DOWN` after every
successful fission for exactly:

`L in {1, 2, 4, 8, 16, 32, 60}`.

The post-fission edit at pulse boundary L is part of the pulse. The composition
immediately after that last edit is the written anchor. Turn the controller off
and run sixty additional untreated fissions using the continuing simulation RNG.
No callback is invoked after release.

For a fixed candidate, matrix, and replicate, pulse length is excluded from the
future seed. These are common random streams, not identical realized futures.

Track during release:

- unrounded cosine similarity to the written anchor;
- frozen risk;
- strict inheritance (`H > 0.9`);
- entropy, top-one abundance share, occupied types, catalytic throughput, growth
  updates, and survival.

Persistence is the first post-release fission at which anchor similarity is
strictly below 0.7. A lineage that completes F60 without crossing is assigned the
right-censor cap 61. A lineage that becomes incomplete before crossing is assigned
the first unobserved registered boundary (`observed + 1`) as an adverse loss.

For every matrix, average persistence over its six replicates at each L and
compute Spearman correlation across the seven pulse lengths. The primary
accumulating-hysteresis gate requires the 95% whole-matrix bootstrap lower bound
of mean matrix Spearman to exceed zero separately in both candidates. This is the
only confirmatory efficacy gate for CR9.

## CR9-B: periodic feedback

Use an independent future seed domain and run sixty fissions under:

- `MODEL_EVERY_1`, `MODEL_EVERY_2`, `MODEL_EVERY_4`, `MODEL_EVERY_8`, and
  `MODEL_EVERY_16`;
- budget-matched `RANDOM_EVERY_1`, `RANDOM_EVERY_2`, `RANDOM_EVERY_4`,
  `RANDOM_EVERY_8`, and `RANDOM_EVERY_16`;
- `NOOP`.

For period K, the action occurs after successful fissions K, 2K, 3K, and so on.
Random actions are uniform over all then-legal substitutions and use a separate,
policy-specific action stream. They never consume the future simulation stream.
The model and corresponding random policy therefore share the intended action
budget; realized counts can differ only when their diverged trajectories end
early, which is retained and reported.

Report inherited-boundary fraction, fixed-horizon adverse inheritance (missing
registered boundaries count as non-inherited), edits, breaks, nonoverlapping
break-and-run3 episodes, longest inherited run, survival, frozen risk, entropy,
concentration, occupied types, throughput, and growth updates.

For each registered period and candidate, report model minus budget-matched
random and model minus `NOOP` matrix-bootstrap intervals and paired sign
randomizations. Define the descriptive minimum-feedback interval as the largest K
whose 95% lower bound is positive against both controls. This criterion is
reported but cannot rescue a failed pulse-ladder gate.

## CR9-C: event-triggered feedback

Use a third independent future domain and run sixty fissions under:

- `THRESHOLD_015`;
- `THRESHOLD_025`;
- `THRESHOLD_035`;
- `CONTINUOUS`; and
- `NOOP`.

After every successful fission, score the unedited state with the frozen
predictor. A threshold policy applies `MODEL_DOWN` if and only if risk is strictly
greater than its registered threshold. `CONTINUOUS` applies `MODEL_DOWN` after
every successful fission. `NOOP` never edits. Thresholds are not tuned or selected
using CR9 outcomes.

Report inherited-boundary fraction, fixed-horizon adverse inheritance, edits per
sixty registered fissions, edits per inherited boundary, pre-action threshold
excursions, post-action residual excursions, survival, and the same state/process
outcomes as the periodic experiment. For every threshold, report inheritance
gain over `NOOP`, edit savings relative to `CONTINUOUS`, and the fraction of the
continuous inheritance gain recovered. No single threshold is selected as a new
registered controller after results are seen.

## Randomness, inference, replay, and integrity

Purpose-separated seeds are frozen for matrix generation, initial composition,
main trajectory, pulse futures, periodic futures, periodic random actions, event
futures, bootstrap, randomization, replay, and smoke validation. Policy identity
is absent from all future seed keys. Action randomness is absent from simulation
streams.

Use 4,096 whole-matrix bootstrap draws and 4,096 paired whole-matrix sign
randomizations. The directional improvement tests use positive one-sided paired
sign randomizations. Pairwise p-values are Holm adjusted within each registered
periodic or event-triggered contrast family. Candidates are never pooled to
rescue disagreement.

Every pulse, periodic, and event-triggered future is regenerated in a complete
second replay. `NOOP` callbacks are checked against the plain simulator. Release
mode invokes no callback and applies exactly zero interventions. Machine-readable
lineage, action, matrix, inference, state, and trajectory artifacts undergo exact
readback checks.

## Claim boundaries and stop rule

A positive pulse result supports accumulating transient hysteresis under longer
active steering. Periodic and triggered results can identify more economical
active-maintenance schedules. They do not support an installed compotype,
autonomous agency, biological memory, error correction, life, real prebiotic
chemistry, a universal origin-of-life mechanism, strict-eight control, or
Phi/PhiID intervention.

CR9 seals its complete result and stops. CR10 is not launched automatically.
