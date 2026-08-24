# Registered exploratory dynamic-regime protocol

## Primary question

Does a mass-preserving one-molecule difference contract or expand under
matched environmental randomness, and does the sign change across a controlled
surface of catalytic strength and turnover?  The original GARD setting is
retained as an explicitly marked point on that surface.

Fresh development and untouched confirmation matrices are generated from new,
disjoint seed domains.  They are not conditioned on an earlier PH outcome.
Both reconstructed simulator candidates are required throughout.

## Dynamic surface and coupled twins

Each realized catalytic matrix is multiplied by
`{0.125,0.25,0.5,1,2,4,8}` and the leave rate by `{0.5,1,2}`.  The original
point is `(1,1)`.  After 64 fissions of burn-in, one molecule token is moved to
a different type.  The original and perturbed compositions then run for 32
fissions with shared Poisson events and shared molecule-token priorities.
Identical states must remain bitwise identical.

Total-variation damage is primary.  The finite-size exponent is
`log((D8+1/160)/(D0+1/160))/8`; cosine distance, coalescence, survival,
integrated damage, and saturation are corroborating readouts.  A deterministic
expected-flow fixed point and its simplex-tangent stability margin provide an
independent diagnostic.

Development selects a shared ordered point (minimum exponent), expansive point
(maximum), and boundary point (smallest absolute exponent among cells with
above-median integrated damage).  Confirmation still evaluates all 21 cells.

## PH and frozen-carrier overlays

At the selected three points, plus the original point when distinct, fresh
32-generation lineages measure the weaker F12 break-and-run endpoint, strict
eight, break timing, terminal clusters, and within- versus cross-lineage
similarity.  F12 and strict eight are always reported separately.

The two carrier settings frozen in the completed carrier campaign are run
without retuning on its 47-rule confirmation bank at ordered, boundary, and
expansive points.  Correct, zero, and shuffled carriers share future noise.
This asks whether carrier efficacy is regime-dependent; it cannot rescue or
replace the prior failed constructive-memory gate.

## Gates

A boundary requires an ordered 95% interval below zero, an expansive interval
above zero, an adjacent sign crossing, and an integrated-damage peak within one
grid step, in both candidates.  “Edge of chaos” additionally requires the
mean-field stability association to agree in both candidates.  The original
point is boundary-adjacent only if it is within one grid step and its 90%
exponent interval is wholly inside `[-0.02,0.02]` in both candidates.

F12 and strict-eight enrichment each require the whole-matrix bootstrap lower
bound for boundary minus the mean of the two flanks to exceed zero in both
candidates.  Carrier enrichment additionally requires positive boundary
correct-minus-zero gain, positive difference-in-differences against both
flanks, and above-chance multiform decoding.

## Runtime and audit

A production-seed-safe benchmark selects the largest registered A/B/C tier
projected inside six hours, reserving time for full replay and reporting.  The
detached campaign soft-stops at 7.5 cumulative hours and is hard-killed at 8
hours.  Restarts cannot reset the ledger.  Every scientific checkpoint is
recomputed exactly before a complete verdict.

