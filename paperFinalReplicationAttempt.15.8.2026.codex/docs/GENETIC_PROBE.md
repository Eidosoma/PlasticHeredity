# Genetic probe for omitted settings

## Purpose and boundary

The `probe` command searches method choices that the manuscript does not
report. It is deliberately separate from the registered replication. Matching
published aggregates after seeing them is calibration, not independent
confirmation; the untouched holdout seed cohort is the generalization check.

The paper-reported model dimensions remain fixed: 100 molecular types, initial
size 40, fission size 80, 100 generations, and the reported lognormal catalytic
matrix. The genome varies only unresolved kinetic exposures, preprocessing,
causal-estimator, compotype, and intervention choices.

## What each paper figure contributes

| Figure | Readable constraint | Use in the probe |
|---|---|---|
| 1 | Schematic growth, fission, compotype detection, and Φ workflow | Structural implementation check; no numeric target. |
| 2 | Aggregate Φ trend p=0.1995; trajectories span roughly 800–1300 molecular steps | Molecular-step scale is fitted. The p-value is reported as a diagnostic, not fitted, because it depends strongly on cohort size. |
| 3 | Mean Spearman ρ=0.139; 73% positive and 54% positive/significant; negative/significant bar about 5% | Fitness targets. The 5% value is explicitly marked as digitized. |
| 4 | Replicating-state Φ is higher in 57% of runs; plotted medians are approximately 0.20 (drift) and 0.85 (replicating) | Fitness targets; plot-derived medians have wide tolerances. |
| 5 | Median prediction accuracies are approximately 84.5% (Φ), 80.5% (composition change), 79.5% (compositions), 79.0% (fluxes), and 61.0% (dummy) | Validation-only on the winning genome. These digitized values never influence genetic fitness. |
| 6 | Table 1 outcomes and plotted probability slopes (max 0.041, control 0.008, min −0.030 percentage points/generation) | Full-objective fitness targets. Slopes are converted to probability units. |

The prose also supplies the spike-time and spike-distance correlations (ρ=0.66
and 0.71), Ljung–Box fractions, and Table 1 values. Exact targets, tolerances,
weights, and provenance are written to every result directory.

## Genome

| Gene | Bounds or choices |
|---|---|
| log10 join exposure | −5.2 to −3.5 |
| log10 leave exposure | −5.5 to −3.2 |
| retain zero-event updates | true / false |
| log10 CLR pseudocount | −3.0 to 0.3 |
| spectral partition cut | zero / median |
| causal measure | WMS / MMI synergy |
| log10 covariance ridge | −10 to −5 |
| compotype similarity cutoff | 0.30 to 0.995 |
| minimum recurrences | 2 to 12 |
| reference states | generation ends / all states |
| similarity | cosine / simplex Euclidean |
| recurring-composition reference | medoid / neighbor centroid |
| intervention estimator (`full` only) | leakage-free online initial / matched control |

Join and leave *exposures* are searched rather than separately varying kinetic
constants and tau, which are not identifiable from a Poisson leap trajectory.

## Fitness, seeds, and safety

Each candidate is run on the same deterministic calibration seeds. Fitness is
the weighted mean pseudo-Huber distance from paper targets after dividing each
residual by its documented tolerance. This is robust to a single badly matched
target. Failed candidates and traces over the configured step cap receive a
large finite penalty rather than crashing the search.

Elitism, tournament selection, bounded mutation, and mixed continuous/
categorical crossover generate later populations. Every evaluation is cached
by evaluator version, genome, fixed configuration, seed cohort, and diagnostic
mode. `checkpoint.json` is written after every generation, so rerunning the
same command resumes safely. `runtime.json` records the live PID and status.

The winner is evaluated once on untouched holdout seeds. Figure 5 forecasting
is then run on at most 100 of those holdout runs, matching the manuscript's
sample size, and remains outside fitness.

## Commands

A short rehearsal:

```bash
MPLCONFIGDIR=/tmp/aor-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/aor-replicate probe \
  --output results/probe-rehearsal --population 16 --ga-generations 4 \
  --calibration-runs 16 --holdout-runs 32 --workers 8 --objective full
```

The checked-in overnight profile uses 64 candidates, 40 genetic generations,
64 calibration seeds, and 256 untouched holdout seeds:

```bash
./scripts/run_probe_overnight.sh results/probe-overnight-full
```

On the machine used to build this repository, extrapolation from the
paper-dimensional rehearsal gives roughly six to seven hours. Candidate trace
lengths can move that estimate. Eight workers leave two logical CPUs free.

Key outputs are `generation_history.csv`, `top_candidates.csv`,
`best_candidate.json`, `SUMMARY.md`, `convergence.png`, `checkpoint.json`, and
the persistent `cache/` directory.
