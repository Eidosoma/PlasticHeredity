# Strict-event geometry audit

This isolated post-hoc audit investigates why the registered strict coherent-eight event is sensitive to the compositional similarity metric and whether a model trained for a Bray–Curtis event can recover held-out state information that a cosine-trained transfer model misses.

It does not edit the preprint. All new files remain below this directory. The simulation inputs and archived development/confirmation cohorts are read-only.

## Scientific design

- Replay the same deterministic 32-fission futures for all retained REGDEV and REGCONF natural states and all 128 registered branches.
- Score three endpoints with the same break → later inherited run of eight → mutual coherence → old-anchor separation shape:
  - registered cosine cutoffs;
  - the prior globally percentile-mapped Bray–Curtis cutoffs;
  - new relation-specific Bray–Curtis cutoffs.
- Calibrate the last set only on 16 fixed branches per REGDEV state. Boundary, pairwise-coherence, and anchor comparisons receive separate empirical-CDF mappings. Event prevalence is never used in calibration.
- Record an ordered failure gate for every branch and cross-evaluate the exact qualifying window under the other metrics.
- Characterize every event window using Shannon effective species number, occupied types, leading-species shares, compositional and occupied-set turnover, growth updates, and metric margins.
- Match each event without replacement to a same-state negative branch that reached an inherited run-8 precursor.
- Refit the original candidate-separated no-PCA nested offset-ridge suite for each development endpoint, seal it before relation-specific confirmation scoring, and compare `h10_state` with `h10` on held-out confirmation halves.
- Use 4,096 matrix bootstraps and 4,096 matrix-block randomizations. Holm adjustment covers the four target-matched candidate-by-half cells within each endpoint.

This is a robustness and predictive-diagnosis analysis, not a causal intervention or a new prospective confirmation. Every endpoint follows one selected daughter at each fission; it is not a both-daughters fidelity test.

## Run

From this directory, using the replication environment:

```bash
/home/robert/Projects/replications/PlasticHeredity/replicators.13.8.2026.codex/.venv/bin/python run_analysis.py all --workers 14
```

The run is checkpointed by natural state. It can be safely restarted with the same command. Individual stages are:

```text
prepare
calibrate --workers N
replay-development --workers N
fit-seal
replay-confirmation --workers N
analyze
report
verify
status
```

`prepare` freezes source identities and the scientific protocol before the new cutoffs or target-specific model results are read. `fit-seal` is deliberately required before confirmation replay. `status` is read-only and reports valid checkpoint counts.

## Outputs

- `artifacts/protocol/analysis_protocol.json`: frozen design and input identities.
- `artifacts/calibration/`: development-only relation-specific mapping and paired comparison distributions.
- `artifacts/replays/`: assembled development and confirmation scores.
- `artifacts/models/`: target-specific portable models and pre-confirmation seal.
- `artifacts/output/`: tables, figures, final audits, manifest, and checksums.
- `RESULTS_REPORT.md`: technical, data-driven interpretation.
- `LAY_SUMMARY.md`: nontechnical summary.
- `SUGGESTED_TEXT.md`: possible manuscript additions; not applied.

The large per-state checkpoint files live in `artifacts/work/` and are resumable intermediate products.

