# White-room replication plan — Pigozzi & Levin, arXiv 2607.28250v1

"Causal Architecture Dynamics Prior to Arrival of Self-replicators in a Model of
Catalytic Networks Relevant to Origin-of-Life"

**Clean-room rules:** No code by the paper's authors (none is published; we also do not
look at their other repositories). Classic GARD literature/software (Lancet lab: Segré,
Ben-Eli, Markovitch, Shenhav) and third-party ΦID tooling (Mediano/Rosas/Luppi, Toker &
Sommer) are allowed. Everything else is implemented from the paper's Materials & Methods
plus the cited primary literature.

## Target claims to reproduce

| # | Claim (paper) | Criterion |
|---|---------------|-----------|
| C1 | No aggregate trend in Φ_r over molecular time across 100 runs (lin. reg. p≈0.20) | qualitative: non-significant aggregate slope |
| C2 | Punctuated Φ_r spikes (>3 SD above mean) in most individual runs | spikes present in majority of runs |
| C3 | Φ_r positively correlates with being in a self-replicator state: ~73/100 runs positive, ~54 significant (Spearman); Φ_r higher during self-replication in ~57/100 (Mann-Whitney p<0.001; Fisher combined p<0.001) | majority positive & significant; Fisher combined p<0.05 |
| C4 | Φ_r trajectories have temporal structure: Ljung-Box rejects white noise ~86/100; 100/100 after differencing | majority reject |
| C5 | MLP fed first 25% of Φ_r trajectory predicts remaining 75% of self-replication trajectory better than baselines (Δcomposition, raw counts, fluxes, dummy), Mann-Whitney p<0.01 | Φ_r model beats all baselines |
| C6 | Spike timing correlates with self-replication probability (ρ≈0.66) and inter-spike distance (ρ≈0.71); spike height n.s. | same sign/significance pattern |
| C7 | Interventions (exhaustive single add/delete after each fission) maximizing Φ_r increase persistence (874±233 vs control 716±198) & consistency (0.52 vs 0.38); minimizing Φ_r worsens all four properties (persistence 559±99, probability 80% vs 88%) | ordering max > control > min on persistence; min worse on probability |

## Model spec (from paper §Materials and Methods)

- GARD, original formulation (Segré, Ben-Eli & Lancet, PNAS 2000).
- N_g=100 molecule types; n_min=40 (initial types sampled uniformly w/o replacement);
  n_max=80; n_gen=100 generations; maxsteps=1000 per generation.
- β catalytic matrix ~ lognormal, mean A=−4, sd σ=4 (parameterization to be pinned via
  PNAS 2000 — see NOTES.md).
- Per generation: stochastic (Poisson) updates of joins/leaves per molecule type until
  n_max or maxsteps; then binomial(0.5) fission; continue with one daughter.
- 100 independent runs (seeds 0..99).

## Φ_r pipeline (from paper + Rosas 2020 / ΦID literature)

1. Substrate: relative compositions (N_g × n_tot per run).
2. Centered log-ratio transform; drop last component (full rank).
3. Minimum-information bipartition → 2 components.
4. Φ_r = I(X_t; X_{t+1}) − Σ_i I(X_t^i; X_{t+1}) (Gaussian estimator), computed as a
   trajectory (sliding window — window length is a free parameter, not stated in paper).

## Phases

- [x] P0: env setup (.venv, numpy/scipy/sklearn/statsmodels/matplotlib/pandas)
- [ ] P1: research reports (GARD spec; Φ_r/ΦID practical recipe) — agents running
- [ ] P2: GARD simulator + composome/self-replicator detector + unit tests
      (validation: recover PNAS 2000 qualitative behavior — composomes emerge,
      compositional similarity across generations)
- [ ] P3: Φ_r pipeline + tests (validation: synthetic systems with known synergy)
- [ ] P4: run 100 sims, compute Φ_r trajectories, save to results/
- [ ] P5: analyses C1–C4, C6 (stats + figures)
- [ ] P6: ML prediction experiment C5
- [ ] P7: intervention experiment C7 (max/min/control × 100 runs)
- [ ] P8: replication report (what reproduced, what didn't, deviations)

## Known underspecified details (decisions logged in NOTES.md)

- Sliding-window length/stride for Φ_r trajectory; MI estimator details.
- MIB search method for 99-dim system (exhaustive is 2^98 — must approximate).
- Self-replicator similarity threshold value.
- Rate constants k_f, k_b and environmental ρ_i (from PNAS 2000, not restated in paper).
- How Φ_r is evaluated for candidate interventions (composition is changed at fission;
  Φ_r needs a time window — presumably recomputed on window including modified state).
- MLP architecture/hyperparameters.
