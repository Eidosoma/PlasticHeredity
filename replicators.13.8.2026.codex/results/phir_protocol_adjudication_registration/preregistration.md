# Chapter 5 protocol-adjudication bridge (PAB24)

Status: prospective Codex registration, to be sealed before scientific generation.

This is a post-clean-room adjudication experiment. It follows a completed read-only
source audit of the independent Fable implementation, so it is not described as a
clean-room replication. It does not overwrite the sealed Codex Chapter 5 pilot,
confirmation, window bridge, or feedback-dose results.

## Question and frozen boundary

Both clean rooms find that feedback changes compositional heredity. They disagree
about which Phi-r formulation moves with that change: Fable's revised nine-atom
measure rises under stabilization, whereas the Codex revised measure falls and the
Codex full-dimensional typeset measure rises. A matched synthetic-input audit found
the revised estimator implementations numerically identical. PAB24 asks which
remaining protocol choice explains the disagreement: launch maturity, controller
selector, observation encoding, or a residual simulator/trajectory effect.

The frozen JOINT_BREAK_RUN3 predictor, both Codex simulator candidates, all Phi-r
estimators, thresholds, and preprocessing remain unchanged. No Fable code, matrices,
states, models, seeds, or result files are imported by the scientific runner.

## PAB-R: archived Codex replay and remeasurement

Before fresh science, the runner deterministically regenerates the sealed D24 arms
`STABILIZE_50`, `DESTABILIZE_50`, `STABILIZE_100`, `DESTABILIZE_100`, and `NOOP`.
It must reproduce their archived record, action, observation, and registered-score
values to numerical tolerance. The final 30 controlled fissions are then measured
five ways:

1. `registered_explicit`: growth observations, explicit fission daughters, and all
   edits including the causally inert post-fission-60 edit (replay control).
2. `endpoint_explicit`: the same trace with no post-fission-60 edit observation.
3. `fable_style`: growth observations only; fission plus intervention remains the
   compound jump between adjacent growth epochs.
4. `phase_normalized`: each fission's pre-growth-to-parent path is linearly
   interpolated onto 16 equally spaced phase points before concatenation.
5. `generational`: post-controller states after fissions 30 through 59 followed by
   the unedited daughter after fission 60 (31 observations).

PAB-R changes no outcome and is diagnostic. Failure of exact replay stops PAB24.

## PAB24: prospective 24-matrix factorial

- 24 entirely fresh catalytic matrices, shared between candidates.
- Candidate 02 and candidate 03 are never pooled.
- Two replicate lineages per matrix and candidate.
- Two launch states:
  - `FRESH`: the natural mass-40 generation-zero composition with empty history;
  - `MATURE`: an untreated, naturally evolved generation-60 state.
- Two frozen selectors:
  - `PANEL12`: 12 legal substitutions sampled with replacement at every action;
  - `EXHAUSTIVE`: every legal substitution scored at every action.
- Two directions per launch-selector cell: `STABILIZE` selects the lowest frozen
  risk and `DESTABILIZE` selects the highest frozen risk.
- One `NOOP` lineage per launch.
- Ten arms total per candidate, replicate, and matrix.
- Sixty controlled fissions; edits occur after fissions 1 through 59 only.
- Final-30 heredity and information measures are primary; all-60 heredity is
  secondary.

The PANEL12 stream is independent of the future stream. At a decision, removal is
uniform over currently present types and addition is uniform over all other types.
Sampling is with replacement and ties are resolved lexicographically after score.
The same panel-selection random stream is used for stabilizing and destabilizing
arms at the same matrix, candidate, replicate, launch, and step. Once their states
diverge, identical random draws can map to different legal edits. Exhaustive ties
are resolved identically.

Future simulation seeds omit arm, launch, selector, and direction. Arms therefore
receive common random streams, not necessarily identical realized futures after
their states diverge. Main-path, panel-selection, future, bootstrap,
randomization, smoke, and replay streams are domain separated.

## Measurements

Every complete lineage is scored with `endpoint_explicit`, `fable_style`,
`phase_normalized`, and `generational` encodings using CLR drop-last preprocessing.
All 16 PhiID atoms are retained, together with revised Phi-r, full-dimensional and
macro typeset measures, normalized full typeset, downward causation, emergence,
synergy persistence, active coordinates, and partitions.

Physical diagnostics are final-30 and all-60 inherited fraction, break count,
growth updates per fission, fission and edit composition jumps, entropy, occupied
types, top-one abundance, and catalytic throughput `x^T beta x`.

## Inference and fixed tests

The catalytic matrix is the inference unit. Replicates are averaged within matrix;
candidates and replicates are also reported in four candidate-by-replicate cells.
Major contrasts use 4,096 whole-matrix bootstrap draws and 4,096 paired
whole-matrix sign randomizations. Holm correction is applied across the four cells
within each registered family. A positive gate requires effect above zero, 95%
bootstrap lower bound above zero, and Holm-adjusted one-sided p below 0.05.

1. Direct bridge: on `FRESH + PANEL12 + fable_style`, stabilization minus
   destabilization revised Phi-r is positive in all four cells.
2. Launch moderation: the preceding revised-Phi-r contrast is larger at `FRESH`
   than `MATURE` in all four cells.
3. Encoding moderation: on `FRESH + PANEL12`, the revised contrast under
   `fable_style` exceeds `endpoint_explicit` in all four cells.
4. Selector moderation: `PANEL12` versus `EXHAUSTIVE` is tested two-sided on the
   fresh Fable-style revised contrast; no directional gate is asserted.
5. Clock robustness: signs under `phase_normalized` and `generational` are reported
   without serving as rescue gates.
6. Manipulation validity: stabilization raises final-30 inherited fraction relative
   to destabilization in every launch-selector family and every candidate-replicate
   cell.

The atom decomposition and update-count relationships are descriptive mechanism
analyses. No candidate pooling can rescue a failed cell.

## Classification

- Direct bridge plus launch moderation: launch maturity is supported as a major
  moderator.
- Direct bridge plus encoding moderation but not launch moderation: observation
  encoding is supported as a major moderator.
- Selector moderation without the above: selection strength/search is supported as
  a moderator.
- If matched Codex trajectories remain opposite under all encodings and launch
  states, the remaining disagreement is classified as simulator/trajectory-level.
- Mixed cells remain a bounded unresolved disagreement.

## Validation, replay, storage, and stop rule

Registration requires legality, deterministic ties and PANEL12 selection, arm-free
future seeds, separated action streams, exact no-op behavior, trace-length and
endpoint fixtures, interpolation endpoints, Phi-r identity fixtures, matrix-block
inference fixtures, serialization, and source/model-hash checks. A non-scientific
smoke may reveal no scientific arm effects.

PAB-R and PAB24 each receive a complete deterministic replay. Only compact tables
are retained; raw molecular trajectories are not persisted. Compressed tables,
arrays, logs, and checkpoint work are excluded from git.

The scientific run is detached and uses at most 12 workers. It refuses to launch
with less than 1.5 GB free. After PAB-R and PAB24, execution stops for human review.
No 48-matrix continuation and no simulator-contract port are authorized here.

Estimated resource envelope: 20--30 CPU-hours, approximately 2--4.5 wall-hours on
12 workers, and less than 600 MB of additional retained/checkpoint storage.

## Claim boundary

This experiment can locate a protocol moderator of the Phi-r disagreement. It
cannot make either Phi-r formulation a universal measure of life, consciousness,
agency, or metaphysical organization. The already replicated causal control of
heredity remains distinct from the open question of how to summarize its internal
information dynamics.
