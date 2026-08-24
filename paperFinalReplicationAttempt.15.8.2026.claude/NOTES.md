# Decision & deviation log (white-room replication)

Decisions made where the paper is silent; revisit as research reports land.

## 2026-08-15 (iteration 2 — research reports folded in)

Confirmed from Lancet-lab primary sources (PNAS 2000; Shenhav 2007; GARD10 code;
Markovitch & Krasnogor 2018):
- **Rate law** (PNAS 2000 Eq. 4): dn_i/dt = (k_f ρ_i N − k_b n_i)(1 + (1/N)Σ_j β_ij n_j);
  catalytic factor multiplies BOTH channels; **k_f=1e-2, k_b=1e-5 s⁻¹** (my draft
  k_b=1e-2 was 1000× too high — root cause of the weak inheritance seen in iter 1).
  ρ_i buffered = 1/N_g (later-classic convention; PNAS closed pool ≈ same since
  n_i << n_TOT).
- **A=−4, σ=4 confirmed** as mean/sd of ln β, i.i.d. over all N_g² entries.
- **Stochastic scheme**: PNAS 2000 = Poisson tau-leap, Δt=0.05 s (matches the
  paper's "Poisson updates"); we subdivide dt when expected events > 16 to bound
  fission overshoot (deviation: PNAS used fixed Δt).
- **Fission**: binomial p=0.5 per molecule at N≥80=2·n_min; follow one daughter.
- **Self-replicator detection**: classic pipeline — pre-fission compositions,
  drift filter at mean-neighbor H < **0.9**, cosine k-means (k≤8, silhouette-max)
  → compotypes; dominant compotype centroid; step labeled SR if H≥0.9 to it.
  (Pigozzi's "similarity threshold relative to the most recurring composition".)
- **Φ_r = Rosas 2020's Ψ** with V = whole system, over the MIB: Φ_r =
  I(X_t;X_{t+1}) − I(M1_t;X_{t+1}) − I(M2_t;X_{t+1}) = Synergy − Redundancy
  (can be negative; that is expected behavior, not a bug).
- **Per-step trajectory**: local (pointwise) Gaussian MI values from one fit per
  run (phyid convention) — windowless; mean equals static Φ_r. Windowed variant
  kept as robustness check.
- Validation targets from PNAS 2000 Fig. 4: parent-daughter H ≈ 0.88±0.10 at
  μ=−4; ≈0.65±0.07 at β=0. Iter-2 smoke test: consecutive-generation H median
  0.89–0.94 across seeds 0–3 ✓; composomes present, SR fractions 16–43% ✓.

Open questions still: paper's "persistence" = total SR steps vs mean episode
length (Table 1 scale suggests ntot≈1000 if total; our ntot≈8000 → compare
proportions, or mean episode length); MLP architecture; intervention Φ_r
evaluation window.

## 2026-08-15 (iteration 1)

- **Kinetics**: forward `k_f·ρ_i·N·(1+(β@n)/N)`, backward `k_b·n_i·(1+(β@n)/N)`
  (same catalytic factor both directions, per Segré 2000 recollection — CONFIRM vs
  research report; also pin k_f, k_b, ρ values).
- **Stochastic scheme**: Poisson tau-leap per "molecular step" with dt=1.0
  (paper says "Poisson updates"; dt not stated). Sizes stay bounded (max ~90 with
  n_max=80 check after each step). May switch to adaptive dt if rates explode
  under confirmed constants.
- **Initial assembly**: n_min=40 distinct types (uniform w/o replacement), count 1 each.
- **CLR zeros**: pseudocount +1 per count before normalizing (paper silent on zeros;
  ≥20 of 100 types are always absent, CLR undefined otherwise).
- **Φ_r window**: sliding window 200 steps, stride 10, lag 1, Gaussian MI with
  ridge shrinkage 1e-3 (window length/stride not stated in paper).
- **MIB**: spectral approximation — sign of Fiedler vector of |corr| graph Laplacian
  (exhaustive over 2^98 bipartitions impossible; Toker & Sommer-style approximation).
- **Composome detection**: greedy medoid over pre-fission compositions, cosine
  similarity threshold 0.8 (threshold value to CONFIRM); step is "self-replicating"
  if cosine to dominant centroid ≥ threshold.
- **Apple Accelerate BLAS** emits spurious divide-by-zero/overflow warnings in
  matmul on this machine; values verified finite — suppress with np.seterr in
  scripts, but re-verify finiteness in run_sims.

## 2026-08-15 (iteration 3 — C7 manipulation check, coarse universe)

- **C7 first pass = null, but uninformative**: interventions did not move Φ_r
  itself (manipulation check: control −1.69, maxΦ −1.73, minΦ −1.59; MW p=0.41/
  0.81). A single-molecule nudge every ~74 steps is washed out. Not evidence
  against the paper's causal claim — evidence my forcing was too weak at fine
  granularity.
- Local-value candidate scoring is FIRST-ORDER IDENTICAL to "raise windowed
  Φ_r most" (appending one row to a window changes mean local value by
  (new−mean)/(n+1) → same argmax). So scoring choice is not the issue.
- **Consistency metric decoded**: "Pearson's ρ between consecutive steps" =
  lag-1 autocorrelation of the binary SR trajectory (episode compactness).
  At fine granularity it saturates (~0.93-1.0); another pointer that the
  paper's step scale is ~1000/run (Table 1: persistence 874 ≈ 88% × ~1000).
- **Coarse universe**: dt=0.4, max_events_per_step=24 → ~10 steps/gen,
  ~800-1200 steps/run, matching Table 1 scale. Full pipeline re-run under
  `results/runs_coarse/`, `*_coarse.json`. Intervention window 100 there.
- **SR prevalence gap**: paper implies 88% of steps in SR; classic cosine-H
  gives 32%@0.9 / 47%@0.8 / 58%@0.7 here; generation-level counting doesn't
  close it. Paper's phrase "similar (in Euclidean space)" suggests a different
  similarity metric. C3 verdict robust to threshold (70-82% runs positive).

## 2026-08-15 (iteration 4 — recovery loop, detector calibration)

- **Table 1 reverse-engineering**: their "36±26%" time-to-first must be absolute
  steps (else contradicts 88% prevalence: first SR at 36% of run caps prevalence
  at 64%). persistence 716 ≈ 0.88 × ~813 → their ntot ≈ 800, persistence = total
  SR steps.
- **Detector calibration sweep** (3 references × 2 similarity metrics × 13
  threshold rules × 100 coarse runs): per-run QUANTILE rules uniquely reproduce
  their tiny ±0-3% probability spread (absolute thresholds give ±26%). Quantile
  0.12 → 88% prevalence exactly. Persistence & time-to-first then land near
  targets. Evidence their SR criterion is per-run relative, not absolute.
- **Consistency decoded (probably)**: lag-1 autocorr of the binary labels can't
  go below ~0.87 under any config; but Φ_r-trajectory lag-1 autocorrelation =
  0.58±0.20 in control runs — the only candidate near their 0.38-0.52 band
  (compositions 0.99, deltas 0.11). "Consistency" plausibly = temporal
  coherence of the Φ_r/similarity signal.
- phyid installed; Φ_r variants (windowed, 2-scalar MIB coarse-graining,
  phyid MMI atoms, τ=2/4) being computed for the C5 recovery grid.

## 2026-08-15 (iteration 5 — C5 recovery grid results)

- 24-cell grid (6 Φ_r variants × 2 label rules × 2 architectures): no cell
  strictly beats all four baselines.
- BUT: classic labels + local Φ_r + in64 → Φ_r is the single best model
  (0.618), significantly beating dcomp (p=1e-4) and raw (p=0.0025); under
  balanced accuracy it also beats dummy (p<1e-4) and TIES flux
  (0.5852 vs 0.5851). τ=2 behaves the same.
- Calibrated 88% labels: Φ_r "beats" dcomp/raw/flux on plain accuracy but the
  majority dummy is unbeatable (0.92); balanced accuracy shows all models ≈
  chance there → those wins were imbalance artifacts. Their Fig 5's "binary
  accuracy" with a beatable dummy implies their labels were NOT ~88% SR in the
  ML task, OR accuracy was balanced/per-class — question for authors.
- **C5 verdict upgrade: partial recovery.** Φ_r carries real predictive signal
  (top or tied-top model in the best-faith configs); strict dominance over
  flux never materializes.
- phyid cross-check: my Ψ implementation vs phyid atom decomposition on the
  same 2-scalar systems: r = 0.9998.

## 2026-08-15 (iteration 6 — authors' operational pipeline reconstructed & tested)

Research (their Commun Biol 2025 + RL preprint methods text; no code opened):
their operational pipeline = pairwise lag-1 Gaussian MI matrix → Fiedler-vector
bipartition → average each part to a scalar → 2×2 ΦID atoms via linear solve →
"causal emergence" = downward causation + causal decoupling = str+stx+sty+sts
(emergence capacity) — which is NOT the printed Ψ formula (differs by the
redundancy sum). Their papers demonstrably swap labels between text and captions.

Tested on our coarse universe:
- Emergence-capacity/2-scalar pipeline FAILS their own correlational results:
  C3 positive 16/100 (paper: 73), spikes 25/100, Ljung-Box 70/100, phi
  autocorr 0.05 (their consistency band 0.38-0.52).
- Raw-composition part averaging (no CLR) is mathematically degenerate —
  singular covariance (simplex closure) — so CLR-then-average was the right
  reading, and the negative result stands.
- Conclusion: the GARD paper's figures were most likely produced by the
  PRINTED formula with multivariate parts (our main pipeline, which reproduces
  C2-C4 closely), not the sister papers' scalar coarse-graining. Key question
  for the authors.
- Also: my Ψ estimator cross-validated against phyid (r=0.9998).

## 2026-08-16 (iteration 7 — "try everything": figures rendered, regime found)

- **Figures finally rendered** (poppler installed). Fig 5: Φ_r model ≈85%,
  baselines 79-81%, dummy ≈61%. Fig 2: ntot ≈ 800-1200 ✓ (coarse universe
  right); trajectories piecewise-flat with ±60-150 needles. Fig 3: mean ρ=0.139
  (ours 0.10-0.11 ✓). Fig 6C: SR probability axis 78-88%; control trend FLAT
  (p=0.47) unlike ours. Table 1 time-to-first has "%" — internally inconsistent
  with 88% probability; ML dummy at 61% ⇒ ML labels ~60/40 ≠ Table 1's 88%
  (paper used different labelings in different experiments).
- **Label-shape sweeps** (gen-level, latched, hysteresis): recover the paper's
  significance pattern (Φ_r beats dummy + dcomp + raw in several configs) but
  not the accuracy level; latch overshoots (prevalence 0.97) because our onsets
  are early (~11% vs their ~36%).
- **RF instead of MLP**: no lift; flux/raw favored. Model class not the lever.
- **Per-generation 2-scalar Ψ** (needle-shape hypothesis): C3 49/31 — worse
  than multivariate local (82/77); ranges ±2.7 not ±100. Weakened.
- **C7 rollout scoring**: manipulation check FAILS again (max didn't raise Φ_r,
  p=0.63; min sits higher). Third intervention implementation, still no
  working bidirectional Φ_r steering. C7 unreproduced; evaluation-horizon
  question for authors is critical.
- **REGIME SWEEP (24 cells)**: fate-predictability varies 6× across classic-
  compatible regimes. Winner: tau-leap, k_b=1e-4 (GARD10's value), k_f·ρ=1e-3
  → fate ρ²=0.67, R²=0.61 (current regime: 0.10/0.22). "pred" universe added
  (dt=0.1 → ~1000 steps/run): results/runs_pred. If their accuracies live
  anywhere, it's here — C5 and C7 to be re-run in this regime.

## 2026-08-16 (iteration 8 — endgame: pred universe + leakage discovery)

- **Pred universe C5**: accuracies did NOT rise (0.53-0.58); fate ρ² at proper
  granularity/n=100 is 0.256 (sweep's 0.67 was partly 30-run noise + coarse-
  feature smoothing). No classic-compatible regime reaches their 80% baselines.
- **LEAKAGE DISCOVERY**: leak-free Φ_r (Gaussian fit on first 25% only) drops
  from 0.607 → 0.527, below every baseline (whole-fit > leak-free p=1e-4).
  Whole-run-fit local values leak future info into "early" features; causal
  baselines have no such channel. Windowless local estimation is the
  parsimonious reading of the paper's own trajectory ⇒ their Fig 5 Φ_r
  advantage is plausibly this leakage. Also inflates our earlier "partial
  recovery" C5 cells (phi>dcomp/raw wins were leakage-assisted).
- C3 (contemporaneous association) unaffected by this concern.

## 2026-08-17 (iteration 9 — code-faithful Φ_R tested; C5 verdict stands)

- Context: the sister replication pulled the authors' public repo
  (github.com/pigozzif/PhiRL) — the printed Ψ formula is NOT what their code
  computes. Code quantity = revised Φ_R: the nine PhiID atoms with synergy on
  either side plus both cross-transfers = naive whole−parts with the
  double-counted redundancy added back once; unnormalized; computed on TWO
  macro-averaged Fiedler halves; pointwise Gaussian entropies, pointwise-MMI
  redundancy, Möbius inversion. Port (verified against their code to ~1e-14 in
  the sister repo) added as src/phi_r_code.py; battery src/recovery_phir.py →
  results/recovery_phir.json.
- Coarse universe regenerated (raw npz had been cleaned): C5 baselines match
  recovery_c5.json to 4 decimals ⇒ counts/labels bit-identical; Ψ rows shift
  ~0.01 (sklearn-version KMeans inside the MIB search); C2 spikes 95/100 vs
  recorded 94.
- SIGN REGIME: Φ_R run-mean +0.96±0.20, positive in 100/100 runs; printed Ψ
  −1.23±0.60, negative in 99/100; run-mean Pearson between the two = −0.26.
  Same molecules, two nearly unrelated instruments — the sister-repo finding
  ("one name, two thermometers") holds in the GARD universe too.
- C3 under Φ_R: positive 31/100 (12 sig), higher-in-SR 9/100 — FAILS the
  paper's 73/100; the printed formula still reproduces it (78/100, 64 sig).
  C4: 84/100 (paper 86; Ψ gave 99). Consistency autocorr 0.26±0.19 (paper band
  0.38–0.52; Ψ 0.58). ⇒ in OUR GARD rebuild the paper's correlational figures
  still look like the PRINTED formula, not the repo quantity — the opposite
  adjudication from the sister chemistry, where only the code quantity moved.
- C5 NOT rescued: Φ_R accuracy 0.601/0.604 ≈ Ψ (0.607/0.588); beats dcomp
  (p=0.0006/0.013), grazes raw (p≈0.05), never flux (0.616/0.625) or dummy;
  leak-free Φ_R (ALL estimation — dead-filter, MI matrix, Fiedler parts,
  moments — restricted to the input window) drops to 0.566/0.556, below every
  baseline. The code quantity has the same whole-run-moments leakage channel
  as Ψ (pointwise local entropies use whole-series covariances). Quantile12
  labels: majority dummy 0.92 still unbeatable.
- phyid cross-check of the port on the same macro pairs: r = 0.40–0.92 per run
  (phyid's MMI redundancy is average-level, the code's is pointwise — related,
  not interchangeable).
- Net: C5 ❌ stands under both instruments; the "which instrument produced the
  figures" question is now sharper — in our rebuild NEITHER instrument
  reproduces all the paper's figures, and the correlational core (C3) sides
  with the printed formula.

## 2026-08-17 (iteration 9b — typeset equation inspected: phantom denominator)

- Rendered p.7 of the PDF and inspected the equation visually (not via text
  extraction). The typeset formula is **unnormalized**:
  Φ^r = I(X_t, X_{t+1}) − Σ_i I(X^i_t, X_{t+1}); the "denominator" seen in
  text extraction is the SEPARATE "Where I(X_t,X_{t+1}) is the time-lagged
  …" definition line stacked under the equation by pdftotext — a phantom
  fraction. This folder's unnormalized reading is correct; the sister
  reconstruction's normalized reading traces to this extraction artifact
  (consequence limited: positive per-run scaling, sign/correlations
  unaffected).
- No formatting issue on the load-bearing parts: past term clearly X^i_t,
  future clearly the WHOLE X_{t+1} (no lost superscript), no dropped
  redundancy term. The printed equation really is Ψ-style naive subtraction,
  not Φ_R. Doubled glyphs (𝛷𝛷𝑟𝑟) are Word→PDF extraction cosmetics.
- The printed-vs-code divergence is TEXTUAL, visible on one page: the
  equation follows Rosas [86,87] (Ψ), while the surrounding prose describes
  the code pipeline per Mediano ΦID [89] — "reduce the system to a
  computationally tractable two-component system" (tractability only follows
  if each half is collapsed to a scalar = the unstated macro-averaging) and
  "the ΦID decomposes … into information atoms, one of which is our causal
  emergence as expressed by Φ^r in the equation above" (false for the
  printed equation — Ψ is an 8-atom signed sum, Φ_R a 9-atom sum; the
  sentence only makes sense as a description of the atom-based code).
- Net: the paper carries BOTH definitions interleaved; a reader implements
  the equation, the code implements the prose.

## 2026-08-17 (iteration 10 — authors' repo read; third candidate tested)

- Cloned github.com/pigozzif/PhiRL (public). Contents: RL project only — NO
  GARD simulator, NO SR detection, NO ML prediction, NO intervention code.
  The intervention evaluation-horizon question stays open for the authors.
- Their compute_phi (main.py) returns FIVE quantities per call: "synergy"
  (sts atom), "causation" (stx+sty, downward causation), "redundancy"
  (rtr added to ITSELF — copy-paste bug, doubled), "integrated" =
  local_phi_r (the 9-atom Φ_R; atom set verified against information.py
  directly — matches our port), and "emergence" = synergy+causation =
  sts+stx+sty (3 atoms). The GARD paper says "causal emergence" — so their
  code's own naming makes emergence3 a THIRD figure-candidate besides
  printed Ψ and Φ_R. remove_autocorrelation exists but is DEAD CODE (never
  called): no prewhitening. analysis.py also runs a time-shuffled Φ null
  control the GARD paper never mentions.
- emergence3 + bare sts tested (recovery_emergence3.py →
  results/recovery_emergence3.json), same macro pipeline, coarse universe:
  - emergence3: +0.73±0.16, positive 100/100; C2 spikes 89/100; C3 positive
    36/100 (14 sig), higher-in-SR 8/100 — FAILS paper's 73/100; C4 75/100;
    consistency 0.16±0.12 (band 0.38–0.52).
  - sts alone: −0.12±0.27, positive 31/100; C2 38/100; C3 48/100 (24 sig);
    C4 48/100; consistency 0.10.
- Candidate scoreboard for "what made Figure 3" (C3 positive /100; paper
  says 73): printed Ψ 78 ✓ · sts 48 · emergence3 36 · Φ_R 31. Consistency
  band 0.38–0.52: Ψ 0.58 (closest) · Φ_R 0.26 · e3 0.16 · sts 0.10. C2
  "most runs": Ψ 95 · Φ_R 95 · e3 89 · sts 38. C4 (paper 86): Φ_R 84
  (closest) · e3 75 · Ψ 99 · sts 48.
- Net: every macro-pipeline quantity their code exposes FAILS the paper's
  correlational core in our rebuild; only the printed multivariate Ψ
  reproduces it. The paper's prose describes the code, but its FIGURES
  match the equation. Sharpest form of the question for the authors.

## 2026-08-17 (iteration 10b — closure test, spike route, and the control
## that killed the rescue)

- Battery relaunched with OMP=1 caps + process pool after OpenBLAS thrash
  (first attempt killed at 1/20 rows, >20 min/row; relaunched row 1
  reproduced the killed run's numbers exactly — determinism intact).
  Outputs: results/recovery_lattice_spikes.json, results/spike_control.json,
  feature cache results/lattice_features.pkl.
- ROUTE 1 (full-lattice closure test): all 16 local ΦID atom trajectories as
  MLP features. Leaky 0.633/0.626 — the best trajectory-feature score of the
  entire study, beats dcomp/raw decisively, does NOT beat flux (p=0.08/0.56)
  or dummy. Leak-free 0.608/0.595 — same pattern. ⇒ NO scalar definable on
  the ΦID-macro lattice (any atom weighting: macro-Ψ, Φ_R, emergence3, sts,
  or anything unnamed) can beat the causal baselines as a trajectory feature
  in this universe. C5 is closed for the entire family, not just the
  enumerated variants.
- ROUTE 2 (episode statistics): 9 spike features (counts at 2σ/3σ, max
  excursion, first-spike times, mean gap, above-2σ fraction, window mean/sd)
  of the LEAK-FREE first-25% Φ trajectory. spikes_phir 0.675 — the first
  honest beats-all of the study (p>flux .007, p>dummy .001, classic labels,
  in64); spikes_psi 0.657 also beats all. Briefly looked like a genuine
  relative rescue of C5.
- REPRESENTATION CONTROL (mandatory before claiming rescue): the same 9
  episode statistics computed on the baseline signals. spikes_flux
  0.699/0.697 — beats everything INCLUDING the Φ spikes; spikes_dcomp 0.652
  beats-all at in64. ⇒ the win belongs to the episode REPRESENTATION (9
  clean summaries vs thousands of noisy trajectory bins), not to Φ. Under
  equal representation the ordering is flux > Φ_R-spikes > Ψ-spikes ≈
  dcomp-spikes: Φ never tops flux, again. C5 stays failed.
- Best honest early-fate predictor found all day: spike-stats-of-flux at
  0.70 — still nowhere near the paper's ~85% band (the universe-level
  ceiling stands).
- Cautionary implication for the paper's own Fig 5 design: under the
  paper's protocol (a 1-D Φ feature vs high-dim trajectory baselines), our
  Φ_R spike features would have produced a publishable "Φ_r beats all
  baselines, p<0.01" — dissolved the moment baselines get the same episode
  summary. Representation asymmetry is a second candidate artifact (besides
  leakage) for their Fig 5 ordering. Worth a line in the letter.

## 2026-08-17 (iteration 11 — reviewer battery: zeros fork, C1/C6-Φ_R,
## preregistered C7-Φ_R campaign)

- Plan approved by user: 4-arm C7 (adds random-edit control), full
  100 runs/arm, both universes (fine gets headline only).
  PHIR_C7_PREREGISTRATION.md frozen BEFORE any campaign code.
- **Zeros fork** (zeros_fork.py → results/zeros_fork.json): the paper's
  stated CLR is IMPOSSIBLE verbatim on GARD data — every timestep of every
  coarse run has ≥1 zero-count species (median 0 always-present species per
  run, max 2), so log(0) degenerates every row; even a present-only
  restriction leaves <3 channels. An unstated pseudocount is mathematically
  mandatory. Conclusions are insensitive to which: pseudo 1.0-drop-last vs
  0.5-no-drop give run-mean Pearson 0.92 and the same sign regime
  (+0.96/+0.98, 100/100 positive) and C3 numbers (31 vs 37 /100).
  Preprocessing question for the authors sharpened AND de-fanged.
- **C1/C6 under Φ_R** (run_c1c6_phir.py → results/c1c6_phir.json):
  C1 clean null (slope −1.05e-5, p=0.336) — matches the paper's "no trend"
  BETTER than printed Ψ did (Ψ had per-run + slopes p=0.006). C6
  SIGN-REVERSED: SR-prob vs spike time ρ=−0.36 p<0.001 (paper +0.66);
  vs gap ρ=−0.18 p=0.09 (paper +0.71); height n.s. ✓. emergence3: C1
  fails (spurious + trend p=0.025), C6 gap ρ=−0.30 p=0.007 (also
  reversed). Scoreboard update: Φ_R wins C1, loses C3/C6/consistency;
  Ψ wins C3/consistency, loses C1's strict null; NEITHER matches C6's
  positive geometry — our universe's spike geometry correlates the
  OPPOSITE way under every macro instrument.
- **C7 under Φ_R** (phi_r_point.py + interventions_phir.py; prereg
  results appended to PHIR_C7_PREREGISTRATION.md): equality gate exact
  (0.00e+00); 400 runs in 82 s. **T1 manipulation-check gate FAILED** —
  max vs min realized Φ_R p=0.58 (0.907 vs 0.908; every edited arm incl.
  random sits ~0.05 below control). T2 primaries null; T4 random-arm
  validity PASSES; T5 Ψ unmoved. Frozen-table row 3 fires: C7 untestable
  at one-molecule intervention strength. Manipulation-check failure now
  unanimous across 4 scorer implementations and 2 laboratories.
- Determinism cross-check: control-arm realized Φ_R/Ψ reproduce
  recovery_phir.json's sign-regime values exactly.
- **EXPLORATORY (not preregistered) — between-run polarity test.** The
  sister system's Φ_R result is a between-lineage reading (organized vs
  disorganized lineages under manipulation). Direct GARD analog: run-mean
  Φ_R vs run-level organization across the 100 coarse runs. Result:
  run-mean Φ_R vs SR probability ρ=−0.338 p=0.0006; vs longest-SR-episode
  fraction ρ=−0.304 p=0.002 — significantly NEGATIVE. Run-mean Ψ: +0.23 /
  +0.22 (p≈0.02) — weakly positive. So within one chemical family (both
  systems are β-matrix mutual-catalysis GARD-class), the code gauge reads
  organization with OPPOSITE POLARITY: heredity droplets = organized
  lineages carry MORE Φ_R (fable T1-code result); classic GARD = high-SR
  runs carry LESS. Interpretation: Φ_R measures dynamic integration
  (information crossing the macro cut). GARD self-replication is settling
  into a stationary composome — organization-as-quiescence → less
  cross-half flow; heredity stabilization is active catalytic coordination
  sustained through fissions — organization-as-coordination → more. The
  gauge is real but its polarity w.r.t. "life-likeness" is
  regime-dependent, which constrains any universal-dashboard reading of
  Φ_R (including the sister preprint's "emergence rises with order" —
  true in that regime, reversed in this one).

## 2026-08-17 (iteration 11b — fine-universe Φ_R pass; polarity confirmed)

- Fine universe regenerated (run_sims.py fine; validation: 100/100 Ψ-spike
  runs, matching REPORT's recorded fine column; SR median 0.33).
  run_c1c6_phir.py fine → results/phir_fine.json.
- Φ_R at fine granularity (Δt=0.05, ~7,400 steps/run): sign regime
  +1.96±0.27, positive 100/100 (even more positive than coarse). C1: tiny
  but significant NEGATIVE trend (slope −2.7e-6, p<0.001; ≈−0.02 over a
  run) — the coarse clean null does not survive fine sampling. C2 99/100 ✓.
  **C3 still fails**: positive 39/100 (30 sig), higher-in-SR 28/100 vs
  paper's 73/100 — better than coarse (31) but nowhere near. C4 83/100
  (paper 86 — close). C6: spike-time ρ=−0.21 p=0.04 (still reversed), gap
  +0.14 n.s., height ρ=−0.32 p=0.001 (paper: n.s.). Consistency 0.12
  (band 0.38–0.52 ✗).
- **Polarity inversion CONFIRMED and stronger at fine**: run-mean Φ_R vs
  SR probability ρ=−0.439, p<1e-5 (coarse: −0.338). Robust across both
  granularities: in classic GARD, more self-replication ↔ LESS code-gauge
  Φ_R — opposite to the heredity chemistry. The
  organization-as-quiescence vs organization-as-coordination reading
  (iteration 11) stands.
- Reviewer battery COMPLETE: all plan items executed (zeros fork, C1/C6
  both instruments, preregistered 4-arm C7, fine universe). Under the
  code-faithful instrument the paper's claims now stand at: C1 ✗(fine)/
  ✓(coarse), C2 ✓, C3 ✗, C4 ~✓, C5 ✗ (closed for the whole lattice
  family), C6 ✗ (sign-reversed), C7 untestable (manipulation gate fails).

## Validation so far

- GARD smoke test (seed 0): 12,254 steps/100 gens, sizes bounded, fission ~80-90.
  Consecutive-generation similarity median 0.53 — LOW vs classic composome papers
  (~0.9); likely rate-constant issue. Blocked on research report.
- Φ_r synthetic tests: white noise ≈ 0 (+0.009); independent AR(1) negative (−0.35);
  constructed Gaussian-synergy pair positive (+0.69); unique-driver pair negative
  (−0.77). Estimator + MIB behave as theory predicts.
