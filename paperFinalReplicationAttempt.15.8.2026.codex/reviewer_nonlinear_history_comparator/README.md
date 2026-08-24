# Nonlinear history-only comparator

This isolated reviewer-response analysis asks whether the frozen composite
retains held-out predictive information beyond a reasonably expressive model
of the observable hereditary past.

It fits two history-only challengers:

1. an exactly input-matched nonlinear ridge, consisting of the registered
   direct-history block plus twelve development-fitted principal components of
   a fixed truncated-cubic-spline and pairwise-interaction library; and
2. a shallow histogram gradient-boosted tree using only the registered direct
   variables, with tree size selected by development-matrix-grouped
   cross-validation.

For each cohort and candidate, the lower-development-loss challenger is frozen
as the selected expressive history baseline. Both challengers and the selected
baseline are scored against the same already-observed confirmation branches.

This is a reviewer-prompted post-hoc robustness analysis. It is protocol-locked
before confirmation scoring, but it is not a new preregistered or prospective
confirmation. No new futures are generated.

All writes are confined below this directory.

## Reproduce

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
replicators.13.8.2026.codex/.venv/bin/python \
  paperFinalReplicationAttempt.15.8.2026.codex/reviewer_nonlinear_history_comparator/run_analysis.py \
  all

PYTHONDONTWRITEBYTECODE=1 \
replicators.13.8.2026.codex/.venv/bin/python -m pytest -q \
  paperFinalReplicationAttempt.15.8.2026.codex/reviewer_nonlinear_history_comparator/test_nonlinear_history.py
```

The main report is written to `artifacts/output/RESULTS_REPORT.md`.

