# Protocol amendment 001: exclude manuscript from input-hash enforcement

The initial protocol (`8cfbe951f70d2529c458faa9808d2704346d31f6944509e95a5923933b1cdb64`)
included `PRE_PRINT_PAPER_DRAFT.md` in its read-only source hash list even though
the pipeline never reads the manuscript during replay, fitting, scoring, or
verification.

After freezing, that independently maintained manuscript changed from SHA-256
`ab64045c5ddb1c3ab463f19f4801a5d25f38791ef17d571437ef219d26cb6518` to
`b08b976c759ca9227685fb97dd8f4e0fe2586f927072b33abe63a0fc8d02afaa`.  The
over-broad mutation guard consequently blocked a checkpointed replay restart.

This amendment removes the manuscript from analysis-input hash enforcement and
records it as an excluded non-input.  It does not change any cohort, endpoint,
feature, model, hyperparameter grid, seed, score, inferential family, gate, or
reporting rule.  The amendment occurred before fitting any new model and before
loading any confirmation outcome.  Forty-one main-path replay checkpoints had
been produced; these contain no confirmation branch futures or outcomes.

The original protocol is preserved as `protocol_pre_amendment_001.json`; the
replacement protocol receives a new content-derived identifier.

