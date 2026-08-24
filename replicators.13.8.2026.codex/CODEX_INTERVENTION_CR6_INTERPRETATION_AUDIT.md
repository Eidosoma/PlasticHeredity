# CR6 interpretation audit before CR7

Status: post-result interpretation of the already sealed CR6 campaign. This
document does not modify CR6 data, inference, gates, or claims.

## Preserved registered verdict

CR6 remains a failed *complete registered gate*. Its scientific registration
is `d15ad57c7f925b5aa8585e3ae32090fa08c500b465d657bd0faf84327744b07e`.
The later administrative readback amendment is
`c96d9c305a103736f7063ff29d19e8c16f85487983051d55b9e9396e044c1ef5`;
it changed no scientific value or decision.

The complete gate failed because the preregistered random-versus-no-op
equivalence requirement did not pass separately in every branch half, and the
candidate-02 weak-heterogeneity null interval was not fully contained in the
registered `+/-0.04` margin. No cell, candidate, or regime is pooled to rescue
that verdict.

## What the sealed data do show

Across the three regimes registered to have positive transfer, every one of
the 12 candidate-by-branch-half cells had:

- positive `MODEL_UP - MODEL_DOWN` control;
- a positive 95% whole-matrix bootstrap lower bound; and
- a Holm-adjusted whole-matrix randomization `p < 0.05`.

At the candidate-pooled descriptive level used by the external Fable
comparison, all six positive-regime candidate contrasts retained positive
targeted effects and their random-versus-no-op 90% intervals were inside
`+/-0.025`. This is qualitative evidence for positive parameter-regime
transfer, but it is not substituted for Codex's stricter registered
branch-half gate.

In the weak-heterogeneity predicted-null regime, candidate 03 met the
registered equivalence criterion. Candidate 02 did not: its 90% interval for
`MODEL_UP - MODEL_DOWN` was approximately `[+0.0013, +0.0435]`, narrowly
exceeding the `+0.04` upper margin. The proper conclusion is that the
candidate-02 null is unresolved, not that it is confirmed and not that a
nonzero effect is established.

All primary futures, exact replays, written-artifact readbacks, and checksums
passed. These interpretation statements introduce no rerun, new model,
recalibration, rescue analysis, changed threshold, or changed gate.

## Relation to CR7

CR7 is not authorized by CR6 and will not use CR6 to tune a controller. It is
independently authorized by the previously sealed CR1 model-guided causal
control result and CR3 physical-rule causal control result, as specified in
the full research directive. CR7 uses the original frozen JOINT_BREAK_RUN3
predictor and the already frozen outgoing catalytic-support rule. The CR6
failure remains visible regardless of the CR7 outcome.

