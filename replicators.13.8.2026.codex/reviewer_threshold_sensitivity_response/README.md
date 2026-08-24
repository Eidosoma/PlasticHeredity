# Reviewer threshold-sensitivity response

This directory is an isolated, post-hoc exploratory analysis addressing the
reviewer's endpoint-definition concerns. It writes only below this directory.
Existing manuscript results and the external L36/L37 materials are verified and
read read-only.

## Frozen scope

- F12 family: strict inheritance `H > {0.85, 0.88, 0.90, 0.92, 0.95}`,
  horizons `{8, 10, 12, 16}`, and renewal runs `{2, 3, 4}` (60 cells).
- Strict F32: coupled adjacent/all-pairs `H > {0.88, 0.90, 0.92}`,
  renewal runs `{7, 8, 9}`, and inclusive old-anchor cutoffs
  `H <= {0.80, 0.85, 0.90}` (27 cells).
- Metrics: prevalence, ordinary and matrix-centered branch-half reliability,
  frozen predictor log-loss/Brier/rank gains, and CR1 intervention contrasts.
- Reference distributions: parent-to-selected-daughter H from the exact F12
  replay and a newly simulated two-independent-lineage reference using the
  frozen L36/L37 method (dominant H>=0.90 component centroid for each lineage,
  then cosine H between the two centroids).
- No prediction model or transform is refit or recalibrated. Candidates and
  the original deterministic branch halves remain separate, and every grid
  cell is retained.

F8-F12 values are exact deterministic replay/rescoring. F16 requires a
deterministic continuation from each archived initial state because the F12
terminal states were not retained. The lineage reference is likewise new
contextual simulation, not a reconstruction of missing historical L36 raw
Parquets. Both qualifications are carried into the generated appendix.

## Reproduce

From the repository root, using the repository virtual environment:

```bash
.venv/bin/python reviewer_threshold_sensitivity_response/run_sensitivity.py prepare
.venv/bin/python reviewer_threshold_sensitivity_response/run_sensitivity.py replay --dataset all --workers 14
.venv/bin/python reviewer_threshold_sensitivity_response/run_sensitivity.py analyze
.venv/bin/python reviewer_threshold_sensitivity_response/run_sensitivity.py report
.venv/bin/python reviewer_threshold_sensitivity_response/run_sensitivity.py verify
.venv/bin/python -m pytest -q reviewer_threshold_sensitivity_response/test_sensitivity.py
```

Replay is checkpointed per state under `artifacts/work/`; an interrupted run
can safely be restarted with the same command. Every checkpoint is bound to
the frozen protocol identifier. The `prepare` stage verifies the checksums of
all source result directories and hashes the runner, scoring code, manifests,
and directly referenced L36/L37 files before any sensitivity readout.

## Deliverables

Final files are written under `artifacts/output/`:

- `APPENDIX_THRESHOLD_SENSITIVITY.md`: compact appendix text and baseline
  readbacks;
- `PROPOSED_MANUSCRIPT_AND_REVIEWER_PATCH.md`: proposed insertions without
  modifying the source manuscript;
- four appendix PNG figures;
- complete CSV tables for all F12, F32, CR1, and H-reference results;
- replay, metric-recomputation, protocol, and final verification audits; and
- `SHA256SUMS` manifests.

`REVIEW_AND_PLAN.md` contains the rationale, objection-by-objection response
map, interpretation rules, and fallback language.
