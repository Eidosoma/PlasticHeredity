# Reconstruction specification and assumption ledger

## Clean-room boundary

The implementation was derived from the public `arXiv:2607.28250v1` preprint, mathematical
descriptions in primary GARD literature, and the spectral partition paper cited
by the preprint. The preprint authors' source code and their older codebases were
not consulted. Third-party source was used for scientific definitions only; no
external implementation was copied. NumPy, SciPy, pandas, statsmodels,
scikit-learn, Matplotlib, and seaborn provide general numerical primitives.

The source PDF is not redistributed in this repository. A user-supplied copy is
hashed into each result directory so later analyses can be tied to the exact
manuscript version.

## Parameter ledger

| Component | Choice in this repository | Status and rationale |
|---|---|---|
| Independent runs | 100 | Reported. |
| Molecular types `Ng` | 100 | Reported. |
| Initial size `nmin` | 40 | Reported. Initial types are uniform without replacement, as stated. |
| Catalytic matrix | independent directed 100×100 lognormal matrix per run, log-space mean −4 and SD 4 | Reported distribution parameters. “Mean” is interpreted as the normal mean underlying the lognormal, the conventional parameterization. |
| Generations | 100 | Reported. |
| Fission size `nmax` | 80 | Reported. |
| Maximum updates per generation | 1000 | Reported. |
| Join/leave process | vector Poisson tau-leap using the standard GARD kinetic equation | Poisson updates are reported; the equation is inherited from standard GARD. |
| Basal constants | `kf=0.01`, `kb=0.0001`, environmental concentration `rho=0.01` | Not reported in the preprint; conventional values from published GARD work. |
| Poisson leap duration | `tau=0.5` | Not reported. Registered after calibration to the approximately 800–1300 molecular-step span visible in Figure 2, before testing headline claims. Swept over 0.25, 0.5, and 1.0. |
| Simultaneous events at capacity | sampled events are randomly thinned without replacement | Not reported. Prevents overshooting `nmax` without species-order bias. |
| Zero-event leap | retained as a molecular step | Not reported. Consistent with the paper's “sequence of stochastic updates”; materially affects trajectory length and is documented. |
| Fission | component-wise `Binomial(ni, 0.5)`; one nonempty daughter continues | Distribution reported; empty-daughter rescue and lineage choice are implementation details. |
| Relative compositions | molecule count divided by assembly size | Reported. |
| Zero handling before CLR | additive pseudocount 0.5 | Not reported and mathematically necessary because most type counts are zero. Swept at 0.1, 0.5, and 1.0. |
| CLR rank repair | drop the final transformed type | Reported. |
| Information lag | one molecular step | Implied by the displayed `t` to `t+1` equation. |
| MIP approximation | lagged pairwise Gaussian MI affinity, symmetrized; normalized Laplacian Fiedler sign cut | The paper cites a spectral MIP method but omits exact construction and threshold search. Median Fiedler cut is swept. |
| Partition aggregation | mean CLR value within each side | Not reported. A deterministic, scale-stable two-component projection. |
| Local information estimator | fitted four-variable Gaussian log-density; local WMS = whole-to-future MI minus both part-to-future MIs | The displayed equation is reported, but estimator and localization are not. This construction yields the trajectory required by the figures. |
| ΦID redundancy | WMS is primary; MMI synergy is a sensitivity | The manuscript simultaneously displays WMS and calls the result a ΦID atom, which do not uniquely identify the redundancy functional. |
| Replicator reference | densest threshold-neighborhood medoid among generation-end compositions | Reconstructs “most recurring composition” and GARD compotype practice; exact clustering is not reported. All-step candidates are swept. |
| Composition similarity | cosine/H with cutoff 0.95 and at least three recurring generation ends | Similarity function and cutoff are not reported. The standard tight GARD cutoff is primary; 0.50–0.98 is swept. |
| Intervention actions | every feasible single-molecule `+1` and every feasible present-molecule `−1` | Reconstructs the reported exhaustive additions/deletions. |
| Intervention estimator | fit once at the first scorable fission using only prior states, then hold fixed | The paper does not explain how a distributional time-series measure is evaluated online. This registered choice prevents future leakage and keeps candidate scores comparable. Per-generation historical refitting and completed matched-control fitting remain explicit alternatives in the API. |
| Forecast split | held-out runs: 80% train, 20% test, repeated for 10 deterministic seeds | Reported. |
| Forecast time shape | resample the first 25% of each input to a common grid and predict a resampled binary trajectory over the remaining 75% | The 25/75 task is reported; variable-length handling is not. No future feature enters an input. |
| Forecast model | standardized one-hidden-layer MLP (64 ReLU units), early stopping | MLP is reported; architecture and training settings are not. |
| Dummy | majority label over the training set | Reported concept; training-only estimation avoids test leakage. |

## Statistical reconstruction

- Figure 2 aggregates unequal trajectories without extrapolation. A time point
  is retained while at least 10% of runs contribute. Its median and sample SD
  are regressed on molecular step.
- A positive spike is a local value above the run mean by more than three sample
  SDs, matching the manuscript's textual definition.
- Figure 3 uses within-run Spearman correlation between local causal emergence
  and the aligned binary compotype label. Constant-label runs are recorded as
  not evaluable rather than silently classified as null results.
- Figure 4 uses a one-sided Mann–Whitney test within each run and Fisher's method
  over finite per-run p-values.
- Ljung–Box tests use lag `min(10, floor(n/5))`, because the manuscript does not
  report the lag. Both raw and first-differenced causal trajectories are tested.
- Spike timing analyses use the independent run as the unit. Replication
  probability is correlated with mean normalized spike time, mean normalized
  inter-spike distance, and mean spike height.
- The manuscript's unillustrated comparator claim is reconstructed at the run
  level: mean causal emergence is correlated with scalar graph and nonlinear
  dynamics summaries. The catalytic graph includes ever-observed types and
  edges with `beta > 1`; centrality maxima measure concentration. Nonlinear
  metrics operate on the successive Euclidean change in relative composition.
  This is explicitly secondary because the paper reports no graph threshold,
  centrality reduction, dynamics substrate, temporal window, or multiplicity
  correction. Both raw p-values and Benjamini–Hochberg q-values are emitted.
- Table 1 reports means and sample SDs. Treatment comparisons are two-sided
  Mann–Whitney tests. Figure 6C fits pooled generation-level OLS trends by
  treatment with 95% mean-confidence intervals.

## Known identifiability limits

The manuscript does not provide kinetic constants, stochastic time scale, zero
replacement for CLR, exact spectral MIP construction, ΦID redundancy function,
compotype clustering or threshold, online intervention estimator, Ljung–Box
lag, MLP architecture, time normalization, or random seeds. These are not minor
packaging omissions: several directly determine the headline outcomes.

In particular, the registered 0.95 compotype cutoff does not recover the 88%
control probability in Table 1. A lower cutoff can approach that target, but
selecting it because it matches the paper would be post hoc tuning. The primary
result remains registered at 0.95 and the full curve is emitted alongside it.

Consequently, this project can test whether the claims are robust under a
transparent, standard reconstruction. It cannot establish bit-for-bit or
number-for-number reproduction of an unavailable and under-specified analysis.

Treatment traces are also relabeled across the full 0.50–0.98 cutoff grid.
Those tables report both the manuscript's unpaired Mann–Whitney comparison and
a paired Wilcoxon check that respects the matched seeds; the 0.95 row remains
the registered primary analysis.

The held-out forecasting experiment is likewise repeated across every cutoff
on the exact same control traces and train/test seeds. This separates a failure
of the registered label definition from a failure of predictive information.

## Primary methodological references

- Segré, Ben-Eli, and Lancet (2000), “Compositional genomes: Prebiotic
  information transfer in mutually catalytic noncovalent assemblies,” PNAS,
  <https://doi.org/10.1073/pnas.97.8.4112>.
- Toker and Sommer (2019), “Information integration in large brain networks,”
  PLOS Computational Biology,
  <https://doi.org/10.1371/journal.pcbi.1006807>.
- The public preprint, [arXiv:2607.28250v1](https://arxiv.org/abs/2607.28250v1).
  A user-supplied local copy is the controlling source and is recorded by hash.
