# White-room replication report — arXiv 2607.28250v1

**Target:** Pigozzi & Levin, "Causal Architecture Dynamics Prior to Arrival of
Self-replicators in a Model of Catalytic Networks Relevant to Origin-of-Life."

**Clean-room protocol:** no code by the paper's authors was consulted (none is
published). The GARD model was rebuilt from Lancet-lab primary sources (Segré,
Ben-Eli & Lancet PNAS 2000; Shenhav, Oz & Lancet 2007; Markovitch & Krasnogor
2018; the GARD10 MATLAB reference implementation) and the Φ_r pipeline from
Rosas et al. 2020 (Ψ criterion), Mediano et al. ΦID, and Toker & Sommer 2019
(spectral MIB). All other details come from the target paper's Materials &
Methods.

**Two replication universes.** The paper's parameters imply Poisson updates but
not their coarseness; its Table 1 implies ~1,000 molecular steps/run. We ran
the full pipeline at two granularities: **fine** (Δt=0.05 s as printed in PNAS
2000; ~7,400 steps/run) and **coarse** (Δt=0.4 s; ~1,000 steps/run, matching
the paper's implied scale). Agreement across both = robust.

## Model validation (before any replication claim)

- Kinetics: PNAS 2000 Eq. 4, k_f=1e-2, k_b=1e-5, ρ=1/N_g, ln β ~ N(−4, 4²);
  binomial fission at N=80=2·n_min; N_g=100, 100 generations, 100 runs.
- Consecutive-generation similarity H = 0.91±0.10 at μ=−4 (PNAS 2000 target
  0.88±0.10); 0.61±0.18 at β=0 (target 0.65±0.07); catalysis-graded heredity
  reproduced.
- Φ_r estimator (Gaussian local values, spectral MIB): ≈0 on white noise,
  negative on redundancy-dominated systems, +0.69 on a constructed
  Gaussian-synergy system — as ΦID theory requires.

## Results by claim

| # | Paper's claim | Fine universe | Coarse universe | Verdict |
|---|---------------|--------------|-----------------|---------|
| C1 | No aggregate Φ_r trend (p=0.20) | pooled median: small − trend (survivorship artifact); per-run slopes mildly + (p=0.006) | same pattern | ~ no *robust* trend, but strict null not reproduced |
| C2 | Punctuated >3SD Φ_r spikes in most runs | 100/100 | 94/100 | ✅ |
| C3 | Φ_r–SR correlation + in 73/100 (54 sig.); Φ_r higher in SR 57/100; Fisher p<0.001 | 82/100 (77 sig.); 74/100; p≈0 | **78/100 (64 sig.); 54/100; p≈0** | ✅ (coarse numbers nearly identical to paper's) |
| C4 | Ljung-Box: structure in 86/100; 100/100 differenced | 99/100; 100/100 | 99/100; 100/100 | ✅ |
| C5 | MLP on first 25% of Φ_r beats Δcomp/raw/flux/dummy (p<0.01); Fig 5: Φ_r ≈85%, baselines 79–81%, dummy ≈61% | fails all 5 configs (Φ_r ~0.57; Δcomp/flux ~0.61–0.64) | recovery sprint (17 further configs: label rules, Φ_r variants, RF, regime sweep): Φ_r reaches top/tied-top and beats dummy+Δcomp+raw in several — but **leak-free Φ_r (fit on input window only) drops below every baseline (0.53, p=1e-4)**; whole-run-fit local values leak future info | ❌ — and the paper's own result is plausibly the same leakage (see interpretation) |
| C6 | SR prob. vs spike time ρ=0.66, vs spike gap ρ=0.71; height n.s. | gap +0.52 ✅, height n.s. ✅, time sign definition-dependent | all n.s. | ~ partial (fine) / ✗ (coarse) |
| C7 | maxΦ_r: persistence 874±233 vs control 716±198, consistency 0.52 vs 0.38; minΦ_r worse on all (559±99, 80% vs 88%) | null — but manipulation check failed (interventions didn't move Φ_r) | three intervention implementations tried (immediate local, windowed-equivalent, 10-step rollout with common random numbers): a working bidirectional Φ_r manipulation never materializes (best: max-side only, p=0.015, with SR outcomes unchanged); min-side persistence drops occasionally (p≈0.006–0.11) but cannot be attributed to Φ_r steering | ❌ not reproduced |

## Interpretation

**What replicates robustly:** the correlational core of the paper. In an active
GARD medium, causal emergence (Φ_r) spikes episodically, carries temporal
memory, and is genuinely higher while the assembly occupies a self-replicating
compositional state — across both granularities, robust to the SR-detection
threshold (0.7/0.8/0.9), and surviving a within-generation-phase control the
paper did not run.

**What does not (from the paper's description alone):** the two stronger
claims. (C5) After ~27 configurations spanning label rules (instantaneous,
generation-level, latched, hysteresis, per-run quantile), six Φ_r estimator
variants (incl. a phyid-cross-validated one, r=0.9998), MLP and random-forest
models, and a 24-cell dynamics-regime sweep: Φ_r reaches top or tied-top
predictor and reproduces most of the paper's significance stars — but never
strictly dominates molecular flux, and the paper's ~80% baseline accuracies
are unreachable in any classic-compatible regime (our run fates carry more
irreducible stochasticity). Decisively: the Φ_r feature's apparent edge over
the causal baselines is largely **information leakage** — local Φ_r values are
evaluated under a Gaussian fitted on the whole run, so "early" Φ_r features
carry future information; restricting the fit to the input window drops Φ_r
below every baseline (0.61→0.53, p=1e-4). Since windowless local estimation
is the parsimonious reading of the paper's own per-step trajectory (no window
parameters appear in any of the group's three papers), the paper's Fig. 5
result is plausibly the same artifact — our single most important question
for the authors.
(C7) Even with a verified Φ_r manipulation, steering Φ_r up produced no gain
in self-replicator persistence or probability; only the minimization arm shows
weak effects in the paper's direction. The paper's large intervention effects
(+22% persistence, −22% on minimization) are not reproduced.

## Addendum (2026-08-17): the authors' code computes a different quantity — testing it does not rescue prediction

After this report was written, the authors' public repository
(github.com/pigozzif/PhiRL) became available to the sister replication. Its
`compute_phi` does **not** implement the printed formula: it computes Mediano
et al.'s revised Φ (Φ_R) — the nine ΦID atoms with synergy on either side plus
both cross-transfers, equal to naive whole-minus-parts with the double-counted
redundancy added back once — unnormalized, on **two macro-averaged halves** of
a Fiedler bipartition, with pointwise Gaussian entropies and pointwise-MMI
redundancy. A deterministic port of that pipeline (verified against the
authors' code to ~1e-14 in the sister replication) was added as
`src/phi_r_code.py` and run through this study's battery
(`src/recovery_phir.py` → `results/recovery_phir.json`) on the regenerated
coarse universe (baselines match `recovery_c5.json` to 4 decimals, so
counts/labels are bit-identical to the original batch).

Three results:

1. **Sign regime.** Code-faithful Φ_R is positive in 100/100 runs
   (+0.96±0.20) while the printed Ψ is negative in 99/100 (−1.23±0.60);
   their run-means correlate at −0.26. The printed formula and the shipped
   code are, on this chemistry, two nearly unrelated instruments.
2. **The correlational core sides with the printed formula.** Under Φ_R the
   paper's C3 claim fails in our universe: Φ_R–SR correlation positive in
   31/100 runs (12 significant), higher-in-SR 9/100 — vs the paper's 73/100
   (54 sig.) and our printed-formula 78/100 (64 sig.). C4 lands at 84/100
   (paper: 86) and consistency autocorrelation at 0.26±0.19 (paper band
   0.38–0.52; Ψ: 0.58). In our rebuild, the paper's correlational figures
   still look like the printed formula, not the repository quantity.
3. **C5 is not rescued.** Φ_R's prediction accuracy is statistically
   indistinguishable from Ψ's (0.60 vs 0.61 at in64, classic labels), never
   beats molecular flux or the dummy, and the leak-free control — every
   estimation step (dead-channel filter, MI matrix, Fiedler parts, Gaussian
   moments) restricted to the input window — drops it below every baseline
   (0.60 → 0.57/0.56). The code quantity inherits the same
   future-information leakage channel as the printed formula, because its
   pointwise local entropies are evaluated under whole-series moments.

**Updated verdict:** C5 (❌) and C7 (❌) stand under both instruments. The
open question for the authors is sharpened rather than resolved: in our GARD
rebuild neither the printed formula nor the repository's Φ_R reproduces all
of the paper's figures, and the correlational core (C3) is reproduced only by
the printed formula — the opposite adjudication from the sister replication's
heredity chemistry, where only the repository quantity responded to causal
manipulation.

**Exhaustive C5 rescue attempt (same day; `recovery_lattice_spikes.py`,
`recovery_emergence3.py`, `spike_control.py`).** Four further results close
the C5 question for this universe:

1. *Closure over all scalar definitions*: feeding the MLP all 16 local ΦID
   atom trajectories of the code-faithful macro system — which linearly
   dominates every possible scalar "Φ" on that lattice (printed-Ψ-on-macro,
   Φ_R, the repo's 3-atom "emergence", anything unnamed) — yields the best
   trajectory-feature accuracy of the study (0.633 leaky, 0.608 leak-free)
   but never beats molecular flux or the dummy. No scalar redefinition of
   Φ_r can rescue C5 here.
2. *The repo's other quantities fail C3 too*: their code's own "emergence"
   (sts+stx+sty; +0.73±0.16, positive 100/100) gives C3 positive 36/100 and
   consistency 0.16; the bare synergy atom gives 48/100 and 0.10. Every
   macro-pipeline quantity their code exposes fails the paper's
   correlational core; only the printed multivariate Ψ reproduces it.
3. *Episode statistics briefly look like a rescue*: 9 leak-free spike
   features (spike counts/heights/timings of the first-25% Φ trajectory)
   beat all four of the paper's baselines — the only honest beats-all of
   the study (Φ_R spikes 0.675, p>flux .007; Ψ spikes 0.657).
4. *A representation control dissolves it*: the same 9 episode statistics
   computed on the baseline signals score higher still (flux spikes
   0.699 — beating the Φ spikes themselves). The gain belongs to the
   episode representation, not to Φ; under equal representation flux again
   dominates. This also flags a second candidate artifact (besides
   moment-leakage) for the paper's own Fig. 5 ordering: a 1-D Φ feature
   compared against high-dimensional trajectory baselines wins on feature
   parsimony, not information content.

Best honest early-fate predictor found anywhere in this study: episode
statistics of molecular flux, 0.70 — far below the paper's ~85% band, which
no feature of the early trajectory reaches in any classic-compatible regime.

**Reviewer battery (same day; preregistered where interventional).** Four
final tests complete the code-instrument evaluation:

1. *Zeros fork* (`zeros_fork.py`): the paper's stated CLR preprocessing is
   impossible verbatim — every timestep of every run contains zero-count
   species (median 0 always-present species per run), so log(0) degenerates
   every row and even an always-present restriction leaves <3 channels. An
   unstated pseudocount is mathematically mandatory; happily the choice is
   immaterial (pseudocount 1.0 vs 0.5: run-mean r=0.92, identical
   conclusions).
2. *C1/C6 under Φ_R* (`run_c1c6_phir.py`): C1 is a clean null at coarse
   (p=0.34 — Φ_R's one clear win over the printed formula) but shows a tiny
   significant negative trend at fine (p<0.001). C6 is sign-reversed: SR
   probability vs spike time ρ=−0.36 coarse / −0.21 fine (paper: +0.66);
   gap negative-n.s. (paper: +0.71); at fine, spike height turns
   significantly negative (paper: n.s.).
3. *C7 under Φ_R* (`PHIR_C7_PREREGISTRATION.md`, `phi_r_point.py`,
   `interventions_phir.py`): 4 arms (control/max/min/random-edit) × 100
   matched seeds; scorer verified to 0 error against the port before
   launch. **The preregistered manipulation-check gate failed**: ~88
   opposite-signed Φ_R-directed one-molecule edits per run leave realized
   Φ_R identical between max and min arms (0.907 vs 0.908, p=0.58), with
   every edited arm (random included) slightly below control. SR outcomes
   null; random-arm validity passed. The manipulation-check failure is now
   unanimous across four scorer implementations and two laboratories; the
   paper's ±22% intervention effects require an intervention channel no
   reconstruction has produced.
4. *Between-run polarity (exploratory, both universes)*: run-mean Φ_R
   anti-correlates with run-level self-replication (ρ=−0.34 coarse,
   −0.44 fine, p≤6e-4) while printed Ψ correlates weakly positively. In
   the sister replication's heredity chemistry — the same β-matrix
   GARD-class family — causally stabilized lineages carry MORE Φ_R. The
   code gauge therefore reads "organization" with opposite polarity in the
   two regimes: GARD self-replication is organization-as-quiescence (a
   stationary composome attractor, little information crossing the macro
   cut), heredity stabilization is organization-as-coordination. Φ_R is
   best described as a gauge of dynamic integration whose relationship to
   life-likeness is regime-dependent — a constraint on any universal
   reading of "causal emergence," the paper's included.

**Final per-claim standing under the code-faithful instrument:** C1 ✓
coarse / ✗ fine · C2 ✓ · C3 ✗ · C4 ~✓ · C5 ✗ (closed for the entire
ΦID-lattice family) · C6 ✗ (sign-reversed) · C7 untestable (manipulation
gate fails). Combined with the printed-formula record in the table above,
no single instrument reproduces the paper.

**Irreducible ambiguities** (candidate explanations for the gaps): the paper's
SR-detection similarity ("in Euclidean space") and threshold are unstated —
classic cosine-H at 0.9 yields 32–39% SR prevalence vs their implied 88%; their
Φ_r estimator internals (windowing, MIB search, regularization) are unstated;
their intervention scoring ("which interventions would raise Φ_r") leaves the
evaluation horizon open (ours is first-order equivalent to maximizing windowed
Φ_r); MLP architecture/features are unstated.

## Deviations and judgment calls

Full log in NOTES.md: rate constants and ρ from PNAS 2000 (not restated in the
paper); CLR pseudocount +1; local (windowless) Gaussian Φ_r values, one fit per
run; spectral MIB (exhaustive is 2^98); "consistency" decoded as lag-1
autocorrelation of the binary SR trajectory; intervention window 200 (fine) /
100 (coarse) steps.

## Reproducibility

`src/`: gard.py, composomes.py, phi.py, run_sims.py, analysis_corr.py,
analysis_ml_grid.py, interventions.py, validate_gard.py; addendum:
phi_r_code.py (port of the authors' repository quantity), recovery_phir.py,
recovery_emergence3.py, recovery_lattice_spikes.py, spike_control.py;
reviewer battery: run_c1c6_phir.py, zeros_fork.py, phi_r_point.py
(pointwise window-fit scorer, equality-gated), interventions_phir.py
(preregistered in PHIR_C7_PREREGISTRATION.md). Python 3.9 venv,
numpy/scipy/sklearn/statsmodels. `run_sims.py [coarse]` → batch;
`analysis_corr.py [coarse]`, `analysis_ml_grid.py [coarse]`,
`interventions.py [coarse]` → results/*.json. Entire study ≈ 15 min on an
Apple-silicon laptop.
