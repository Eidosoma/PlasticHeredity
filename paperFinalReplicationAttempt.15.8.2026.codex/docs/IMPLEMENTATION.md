# Implementation map

| Scientific component | Implementation | Verification |
|---|---|---|
| GARD propensities, Poisson exchange, fission | `src/aor_replication/gard.py` | event feasibility, nonnegative counts, deterministic replay, fission bounds |
| Compositional preprocessing | `src/aor_replication/composition.py` | closure, CLR centering, cosine invariance |
| MIP and local causal trajectory | `src/aor_replication/information.py` | lagged MI behavior, nontrivial bipartition, local/global score consistency |
| Compotype labels and Table 1 metrics | `src/aor_replication/replicators.py` | synthetic recurring-composition trace and metric units |
| Exhaustive interventions | `src/aor_replication/interventions.py` | feasibility, max/min score ordering, online-history use |
| Per-run and aggregate statistics | `src/aor_replication/analysis.py` | exercised in the end-to-end smoke workflow |
| Held-out forecasting | `src/aor_replication/forecast.py` | run-level separation and training-only majority dummy |
| Robustness sweeps | `src/aor_replication/sensitivity.py` | fixed-trace threshold/estimator sweeps and matched-seed tau sweep |
| Checkpointing and provenance | `src/aor_replication/storage.py`, `pipeline.py` | smoke rerun and configuration mismatch protection |
| Figures | `src/aor_replication/plots.py` | headless rendering and visual inspection |

The package deliberately keeps raw simulations separate from analysis. A saved
trace contains counts, generation and phase identifiers, joins, leaves,
intervention decisions and scores, the exact catalytic matrix, and the seed.
This makes relabeling and estimator sensitivity possible without rerunning the
dynamics.

Randomness is split into independent beta, initial-composition, exchange, and
fission streams using NumPy `SeedSequence`. Matched treatments use the same seed
and catalytic matrix. Once state-dependent trajectories diverge they consume
the stochastic stream differently, as expected for matched stochastic
interventions rather than forced common-event paths.

