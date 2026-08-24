# Reviewer sequence-history baseline response

This directory contains an isolated, reviewer-prompted post-hoc rescore of the
frozen F12 predictor.  It asks whether the frozen composite remains better than
history models that use the ordered pre-launch inheritance sequence.  It does
not alter the manuscript, existing models, existing results, or either
clean-room implementation.

All writes are confined below this directory.  Confirmation futures are never
simulated: the pipeline deterministically replays natural main paths to recover
pre-launch histories, then scores the already-retained confirmation outcomes.

## Frozen analysis

- Primary cohorts: Codex `scaled5` (200 development/200 confirmation matrices)
  and Fable v2 (1,000 development/200 fresh confirmation matrices).
- Secondary cohorts: the matched 40-matrix headline cohorts in both clean
  rooms.
- Candidates `02` and `03`, and confirmation halves `A` and `B`, stay
  separate.
- Models: first-order Markov, duration-aware semi-Markov (`1,2,3,4,5+`), and
  a direct-history ridge augmented with ordered continuous H, strict-H flags,
  and padding masks.
- The lagged ridge selects lag length `{5,10,20,40,100}` and
  `C={0.01,0.1,1,10}` by five-fold development-matrix-grouped CV.
- The primary inferential family is composite versus lagged history in the
  eight primary implementation/candidate/half cells, Holm-adjusted.
- The analysis is post-hoc and cannot convert the retained cohorts into new
  prospective confirmations.

The originating L53/L54 comparison is not rescored because the required
machine-readable state, prediction, and branch artifacts are absent from this
checkout.

## Reproduce

From `replicators.13.8.2026.codex/`:

```bash
.venv/bin/python reviewer_sequence_history_response/run_analysis.py prepare
.venv/bin/python reviewer_sequence_history_response/run_analysis.py replay --dataset all --workers 12
.venv/bin/python reviewer_sequence_history_response/run_analysis.py fit
.venv/bin/python reviewer_sequence_history_response/run_analysis.py analyze
.venv/bin/python reviewer_sequence_history_response/run_analysis.py report
.venv/bin/python reviewer_sequence_history_response/run_analysis.py verify
.venv/bin/python -m pytest -q reviewer_sequence_history_response/test_sequence_history.py
```

Replay checkpoints are written per matrix and can be resumed.  `prepare` never
overwrites an existing protocol; it verifies that the existing frozen protocol
matches the current source contract.

Outputs are written under `artifacts/output/`, including the scientific report,
cell-level scores, inference table, model-selection audit, proposed manuscript
language, replay audit, figures, and checksums.

