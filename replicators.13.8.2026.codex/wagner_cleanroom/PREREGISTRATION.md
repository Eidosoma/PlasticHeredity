# Wagner clean-room campaign preregistration

Status: frozen before the first non-discarded run. The canonical machine-readable
contracts are `protocols/primary-v1.json` and `protocols/predictor-v1.json`.

## Primary claim

The independent unit is a fresh eligible Wagner rulebook. The registered cohort
contains 240 rulebooks and exactly 3,194,880 futures. No failed proposal, future,
or rulebook is excluded after its outcome is known. A destination is the first
active deterministic point attractor occupied for three consecutive descendant
boundaries; A, B, other point, cycle, and nonconvergence are distinct classes.

For each rulebook, target-risk gain is

`mean(match | state transplant) - mean(match | reset)`.

History crossover is one half of

`P(A | A-history) - P(A | B-history) + P(B | B-history) - P(B | A-history)`.

The destination-prediction contrast is the evaluation-half log loss of a pooled
rulebook committor minus that of a history-conditioned committor. Both use
Dirichlet-0.5 smoothing; the fixed halves are reversed and averaged. Inference
uses 4,096 rulebook bootstrap resamples and retains the conservative 18-test
simultaneous lower bound.

The outcome is `PASS` only when every gate in the primary JSON passes, including
minimum effects, adjusted positive lower bounds, persistence, shuffled-state and
generation-one controls, reliability, exact self/transplant identity, complete
records, semantic-coordinate uniqueness, checksum verification, and replay.

## Predictor extension

Development uses 96 independent rulebooks; model selection, scaling, and
regularization end there. Evaluation uses 128 untouched rulebooks, five restored
history states per rulebook, and 128 futures per state. Its primary F12 event is a
phenotype break followed by three consecutive exactly inherited adult boundaries
within 12 boundaries. Strict F32 and the fixed threshold grid are secondary.

The frozen models are constant prevalence, nine-variable history-only ridge,
history plus strong basin/sensitivity features, and the full present-state and
regulatory-context ridge. Evidence is called `PROMISING_EXPLORATORY` only if the
full model gains at least 0.02 nats over history with a positive interval, has a
positive interval over the strong structural model, and split-half reliability
is at least 0.80. This label cannot change the primary verdict.

## Operations and failure policy

The discarded four-rulebook benchmark must project the entire campaign under
11.25 hours after a 20% safety multiplier. Scientific work is source-checkpointed
and schedule-invariant. New shards stop at 11.5 hours; 12 hours is a hard campaign
limit. Missing cohorts, a registration mismatch, replay/checksum failure, or an
incomplete deadline stop remains explicit and receives no scientific verdict.

The earlier implementation is never a runtime dependency. Earlier numerical
results may be compared only after this campaign has been sealed.
