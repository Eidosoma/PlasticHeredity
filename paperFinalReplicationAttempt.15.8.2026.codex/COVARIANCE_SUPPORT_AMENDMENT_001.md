# Covariance-support execution amendment 001

**Failed registration:**
`ccbcebdf3fd13c9420c94655cb79365d72d891f8dd22ddf532270d6e578de9c2`

All six registered development trajectories and every in-memory support score
completed under the failed registration. Before any score table or stability
gate was written, the post-run summarizer raised `KeyError: False` because
pandas resolved `scores.mode` to `DataFrame.mode`, its built-in method, rather
than to the column named `"mode"`.

This amendment changes no trajectory seed, support, repeat, pair index,
instrument, transform, covariance rule, stability threshold, or stop rule. It
only replaces pandas attribute-style column access with explicit bracket
access, adds a regression test that exercises a frame containing the `mode`
column, and requires a new source-hashed registration. The six exact trace
checkpoints may be reused after their registered seeds and replay hashes pass.

No numerical support result or gate decision was available when this amendment
was written.
