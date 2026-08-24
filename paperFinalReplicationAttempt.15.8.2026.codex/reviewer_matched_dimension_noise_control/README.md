# Matched-dimension nuisance-PCA control

This isolated reviewer-response analysis tests whether the frozen composite
predictor's advantage can be reproduced by extra fitted dimensions carrying no
correctly aligned state information.

For each retained clean-room cohort and candidate, it keeps the history block,
targets, outcomes, model class, ridge penalty, and fitted input count fixed. It
then reassigns the state-block PCA scores across matrices within the same
generation/landmark, refits the final ridge on development, applies an
independent reassignment at confirmation, and scores the already-retained
outcomes.

The reassignment is an exact row permutation. Consequently, the raw state
block's empirical marginal distribution and covariance—and therefore its
development scaler, PCA basis, component count, and component marginals—remain
unchanged. Only alignment with the row's own history, matrix, and outcome is
removed.

Replicate zero is the fixed primary derangement. Another 31 independently
frozen derangements measure pairing sensitivity. This is a reviewer-prompted
post-hoc rescore, not a new prospective confirmation.

All writes are confined below this directory. No manuscript, canonical model,
or canonical result artifact is modified.

## Reproduce

From the repository root:

```bash
replicators.13.8.2026.codex/.venv/bin/python \
  paperFinalReplicationAttempt.15.8.2026.codex/reviewer_matched_dimension_noise_control/run_analysis.py \
  all --workers 12

replicators.13.8.2026.codex/.venv/bin/python -m pytest -q \
  paperFinalReplicationAttempt.15.8.2026.codex/reviewer_matched_dimension_noise_control/test_nuisance_control.py
```

The main report is written to `artifacts/output/RESULTS_REPORT.md`.

