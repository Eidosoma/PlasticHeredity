# Codex Chapter 5 inserted 24-matrix Phi-r window bridge

Status: prospective registration document; results must be written only below
the source-hashed registration generated from this document.

## Motivation and separation from the completed pilot

The completed Codex Chapter 5 pilot is preserved unchanged under registration
`9bf0cfc0050726d0fb6893cfa6f1789363612b50a6154a399477e3748f7726cf`.
Its 48-matrix confirmation remains unauthorized and locked.  The pilot found
that frozen molecular stabilization increased heredity while the registered
rolling-512 revised Phi-r reading decreased.  An external clean-room audit
then reported that a single pooled multi-fission reading increased on its own
trajectories, while a preliminary rolling remeasurement shifted the effect
downward.  The two implementations also used structurally different
quantities under the informal label "typeset".

This inserted study is a new, separately versioned 24-matrix experiment.  It
does not amend, pool with, rescue, or overwrite the completed pilot.  It tests
the prospectively developed hypothesis that the apparent disagreement is
caused by temporal window construction and repartitioning rather than by arm
order, time direction, or a negated Phi-r definition.

No Fable code, matrices, states, seeds, fitted objects, trajectories, result
objects, or controller objects enter this study.  The external numerical
results motivate the registered comparisons but are not fitting targets.

## Frozen substrate and cohort

- 24 entirely fresh catalytic matrices from a new seed domain.
- Codex candidates 02 and 03, never pooled.
- Two natural replicates per candidate and matrix.
- Natural launch state after 60 untreated fissions, using the existing bounded
  natural-path retry contract.
- The immutable Codex 5x JOINT_BREAK_RUN3 predictor, SHA-256
  `9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af`.
- Two 60-fission feedback arms:
  `MODEL_STABILIZE` chooses the legal one-molecule substitution with minimum
  frozen predicted risk after each fission; `MODEL_DESTABILIZE` chooses the
  maximum.
- Both arms use the same purpose-keyed future stream for a given candidate,
  matrix, and replicate.  Arm identity is absent from that stream key.  Paths
  may consume common random streams differently after their states diverge.
- No scientific matrix, failed path, or adverse lineage is replaced.
- Complete deterministic regeneration into a second checkpoint tree is
  mandatory.

If a controlled lineage ends before fission 60, it remains in the acquisition
table. Missing registered boundaries count as non-inherited for the heredity
validity outcome; information readings requiring a complete registered window
are undefined and their eligible matrix count is reported. No retry or
survivor imputation is allowed.

The heredity validity contrast is stabilizing minus destabilizing inherited
fraction over fissions 31--60.  It must be positive in all four
candidate-by-replicate cells before Phi-r differences are interpreted as a
response to genuinely different hereditary behavior.

## Observation stream

Every molecular observation is an unrounded float64/int64 composition.  The
controlled stream begins with the restored post-fission launch composition.
For every subsequent fission it appends, in order:

1. every growth-update composition;
2. the selected continuing daughter (the fission jump); and
3. the post-substitution composition (the intervention jump).

Thus pooled and rolling estimators receive the identical ordered trajectory.
The pooled fissions-41--60 series begins with the post-intervention boundary
after fission 40 and includes every observation through the boundary after
fission 60.  The fissions-31--60 series is constructed analogously from the
boundary after fission 30.

## Primary CLR/drop-last estimator family

Every score applies the already sealed Codex pipeline: additive 0.5 zero
replacement, closure, CLR, final-coordinate removal, past-window active
coordinate selection, within-window standardization, symmetric lag-one
Gaussian-MI graph, deterministic Fiedler bipartition, macro-averaging of the
two halves, and the public-PhiRL nine-atom revised Phi-r sum.  Intervention
and fission transitions are unmasked.

Four readings are computed on every lineage:

1. `pooled20_clr`: one score on fissions 41--60; one standardization and one
   Fiedler partition.
2. `rolling20_clr`: the mean of rolling last-512-observation scores recorded
   after fissions 41--60; every score is standardized and repartitioned.
3. `pooled30_clr`: one score on fissions 31--60.
4. `rolling30_clr`: the mean of rolling last-512-observation scores recorded
   after fissions 31--60.  This is the completed Codex pilot's temporal
   construction.

All 16 PhiID atoms, revised Phi-r, causation, emergence, and
synergy-persistence are retained for all four readings.

## Registered sensitivities and estimator taxonomy

The same four windows are scored after direct raw-count coordinate
standardization without closure/CLR.  This is explicitly named the
`raw_count` sensitivity; it is not claimed to reproduce an external private
pipeline.

Two quantities formerly called typeset are kept distinct:

- `full_typeset`: whole-system multivariate MI minus each full coordinate
  block's MI with the whole future, using the Fiedler coordinate blocks;
- `macro_typeset`: the same algebra after the two Fiedler halves have first
  been averaged into a two-variable macro system.

Neither can be substituted for the other.  Their levels and intervention
contrasts are registered secondary taxonomy results.  `macro_typeset` is
computed for every revised-Phi-r window.  Because the full-dimensional
log-determinant is substantially more expensive and is not a primary endpoint,
`full_typeset` is computed on both pooled windows and at rolling boundaries 40
and 60, exactly the two finite final-30 checkpoints used by the completed
Codex pilot.  The full-dimensional numerator divided by whole MI remains an
explicitly named normalized-full control, not an external top-k/MIB text
estimator.

For every rolling window, retain its active-coordinate set and Fiedler
partition.  Report label-invariant partition disagreement with the matching
pooled partition on common active coordinates.  This is descriptive evidence
about repartitioning, not a causal mediation analysis.

## Primary estimands and gates

The catalytic matrix is the inference unit.  Within each matrix, candidate,
and replicate, first calculate each arm's lineage reading and then the paired
`MODEL_STABILIZE - MODEL_DESTABILIZE` effect.

Primary families, each evaluated separately in candidate 02 replicate 0,
candidate 02 replicate 1, candidate 03 replicate 0, and candidate 03
replicate 1:

1. **Heredity validity:** the inherited-fraction effect over fissions 31--60
   is positive, its 95% whole-matrix bootstrap lower bound is positive, and
   its Holm-adjusted one-sided sign-randomization p-value is below 0.05 in all
   four cells.
2. **Pooled response:** the `pooled20_clr` revised-Phi-r effect satisfies the
   same positive gate in all four cells.
3. **20-fission window moderation:** for each matrix compute
   `(rolling20 arm effect) - (pooled20 arm effect)`.  It must be negative, its
   95% upper bound must be below zero, and its Holm-adjusted lower-tail
   randomization p-value must be below 0.05 in all four cells.
4. **30-fission window moderation:** apply the identical negative gate to
   `(rolling30 arm effect) - (pooled30 arm effect)`.

A **full sign-reversal classification** additionally requires the pooled20
effect to pass positively and the rolling30 effect to have a negative point
estimate and negative 95% upper bound in all four cells.  This is reported
separately and is not required for the more general window-moderation claim.

The paired moderation estimand is the key result: it compares two estimators
on the exact same trajectories.  A pooled result crossing zero is not called
negative; a rolling result crossing zero is not called equivalent.  No
candidate or replicate may be pooled to rescue disagreement.

## Inference and artifacts

- 4,096 whole-matrix bootstrap draws.
- 4,096 paired whole-matrix sign randomizations.
- Holm correction across the four cells within each registered family.
- Persist matrix-level effects, deterministic inference arrays, lineage
  summaries, rolling-window summaries, selected edits, matrix inputs, source
  hashes, seed domains, and replay audits.
- Persist no raw molecular trajectory; only registered scores, partitions,
  compact compositions at fission boundaries, actions, and digests.
- Atomic checkpoints are written only after a complete matrix unit.

## Validation, execution, and stop rule

Before a scientific matrix exists, validation must cover window boundary
construction, process-stable content hashing, arm stream pairing, legal
controller edits, score identities, macro/full typeset separation,
label-invariant partition comparison, matrix-block inference, simulator trace
parity, replay, source hashes, seed separation, and the locked original
confirmation.  A non-scientific smoke test may exercise I/O without exposing
scientific arm ordering, rates, candidate differences, or effect sizes.

The 24-matrix run must execute detached.  It stops after complete replay,
analysis, sealing, and user review.  It cannot create an authorization for or
launch the original 48-matrix confirmation.

## Claim boundary

A passing window-moderation result may support only that the measured
relationship between revised Phi-r and hereditary stabilization depends on
the prospectively specified temporal estimator in these Codex GARD
contracts.  It does not make either estimator uniquely correct, establish
substrate-independent information integration, turn Phi-r into a controller,
or support consciousness, life, agency, biological memory, a universal
origin-of-life mechanism, real prebiotic chemistry, or a literal
Platonic-space/Ruliad interpretation.
