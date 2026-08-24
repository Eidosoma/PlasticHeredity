# Threshold and alternative-metric sensitivity

This isolated reviewer-response analysis completes the requested robustness
grid:

- inheritance/coherence source thresholds `0.85`, `0.875`, `0.90`, `0.925`,
  and `0.95`;
- F12 renewal runs `2`–`5` and horizons `8`, `10`, `12`, and `16`;
- strict coherent windows `6`, `8`, and `10`;
- old-anchor source thresholds `0.80`, `0.85`, and `0.90`; and
- cosine similarity plus Bray–Curtis similarity on normalized compositions.

The Bray–Curtis thresholds are empirical-percentile matches to the source
cosine thresholds. Numeric cosine cutoffs are not reused for a metric with a
different scale. The calibration uses similarity distributions only, not event
labels or predictions.

This is post-hoc robustness analysis, not a new confirmatory gate. Predictors
are applied unchanged. Compact archives lacking composition vectors are
supplemented only by exact deterministic replay of their already-scored seed
streams—no matrices, states, branches, seeds, model fits, or genuinely new
random futures are added.

All new writes are confined below this directory. The manuscript is not
modified.

## Reproduce

From the workspace root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
replicators.13.8.2026.codex/.venv/bin/python \
  paperFinalReplicationAttempt.15.8.2026.codex/reviewer_threshold_metric_sensitivity_extension/run_analysis.py \
  all --workers 12
```

The replay is checkpointed by state and can safely be resumed with the same
command. Individual stages are `prepare`, `replay-foundation`, `calibrate`,
`replay-f32`, `analyze`, `report`, and `verify`.

Run the synthetic unit checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
replicators.13.8.2026.codex/.venv/bin/python -m pytest -q \
  paperFinalReplicationAttempt.15.8.2026.codex/reviewer_threshold_metric_sensitivity_extension/test_sensitivity.py
```

The main deliverables will be written to `artifacts/output/RESULTS_REPORT.md`,
`SUGGESTED_TEXT.md`, three compact figures, and complete machine-readable CSV
tables.
