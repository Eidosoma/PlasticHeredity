# Phase L — the paper-faithful Φ-r instrument
# (preregistered 2026-08-18, BEFORE any measurement ran)

Question: does Φ-r AS THE PAPER'S TYPESET METHODS DEFINE IT —
ignoring the PhiRL repository entirely — couple to plastic heredity?
Phases I/J/K established: the extracted (phantom-denominator) and
typeset-on-macro-halves readings are unresponsive; the CODE quantity
(revised Φ_R on macro-averaged halves) responds. But no instrument
so far has implemented the paper's page verbatim: the typeset
formula with MULTIVARIATE parts (the page never macro-averages),
CLR + drop-last preprocessing (which the code omits), and no
z-scoring (which the code adds but the page never mentions). Phase L
builds that instrument and asks the causal question once more.

## The instrument (`phir_paper.py`, frozen)

Per lineage, on the per-molecular-step composition trajectory
(the page's substrate, Ng × ntot):
1. Relative compositions; pseudocount 0.5 (registered; the page is
   silent on zeros and log(0) is undefined).
2. Centered log-ratio transform, per time point across components
   (the page's prose reading: "each component as the deviation from
   the system-wide geometric mean").
3. Drop the last component (the page's full-rank fix). No z-scoring,
   no dead-channel masking — neither appears on the page; a ridge
   (1e-6 × trace/dim on each covariance diagonal) handles residual
   near-singularity (registered; unavoidable numerically).
4. Minimum-information bipartition: the page defines it ("the two
   components share the least information") but gives no search
   method; registered choice: spectral (Fiedler) relaxation of the
   min-cut on the INSTANTANEOUS (lag-0) Gaussian MI graph over all
   retained components — instantaneous per the page's wording,
   unlike the repo's lag-1 graph. Both blocks kept MULTIVARIATE.
5. The typeset formula, verbatim and unnormalized:
   Φr = I(X_t; X_{t+1}) − I(A_t; X_{t+1}) − I(B_t; X_{t+1}),
   Gaussian log-determinant estimates, one value per lineage.

## Design

Byte-exact deterministic replay of Phase J's ph_stab, ph_destab,
random, and noop arms (seeds/domain 28 untouched; 4 arms × 48
matrices × 2 reps × 2 candidates = 768 lineages), with per-update
recording extended to fissions 21–60. REPLAY GATE: code-Φ_R
recomputed on the fissions-41–60 sub-record must equal the stored
Phase J values exactly. Primary measurement window: fissions 21–60
(T ≈ 480 steps for candidate 03, ≈ 2,100 for 02 — the longer window
is required because the multivariate estimate needs T well above the
198-dimensional joint; the page itself used whole-simulation
trajectories). Secondary: fissions 41–60 (reported with a small-T
caveat for candidate 03).

## Registered tests (matrix bootstrap, 4,096 draws, seed 23;
## candidates never pooled)

- **L1 (the question):** ph_stab − ph_destab on paper-Φr, primary
  window; CI excluding 0 in both candidates = the paper-faithful
  quantity couples to heredity.
- L2 (specificity): random − noop must span 0.
- Descriptive: arm levels (sign of paper-Φr); correlation with
  code-Φ_R across lineages; secondary-window L1.

## Registered predictions

- Levels: strongly NEGATIVE everywhere (two heavily overlapping
  multivariate halves of one assembly double-count massively;
  the macro-halves typeset adjudication already sat at ≈ −0.03 and
  multivariate parts should deepen it).
- L1 lean: NULL (the typeset formula's structure penalizes exactly
  the redundancy that stabilization creates; every non-code
  instrument so far has been flat) — but this is the registered
  question, two-sided, and a pass in both candidates would mean the
  paper's own page contains a heredity-coupled quantity after all.
- L2: null required.

## Boundary

Phases I–K boundaries carry over. Registered choices forced by the
page's silence: pseudocount, ridge, MIB search method, window.
Sealed: this file, `phir_paper.py`, `run_phir_paper.py`,
`phir_code.py`, `run_phir_confirm.py`, `sim.py`, `cohort.py`
(SHA-256 in `results_phir_paper/SEAL.json`).

---

# RESULTS (appended 2026-08-18; nothing above edited)

Replay 204 s; **REPLAY GATE PASS (0 of 768 mismatches)** — the
campaign is byte-identical to Phase J; only the instrument is new.
Raw units `results_phir_paper/phir_paper_units.pkl`; suite 32/32 at
seal.

## Levels (registered prediction HIT)

Paper-faithful Φr is strongly NEGATIVE everywhere: −8.9 to −11.0
(primary window 21–60), −2.6 to −9.6 (secondary 41–60). Two
overlapping multivariate halves of one assembly double-count
massively, as registered. The instrument is nearly uncorrelated
with the responsive code quantity (Pearson +0.18 / +0.09).

## L1 (the question): NO coherent coupling

- Primary window (21–60): cand 02 +1.2625 [+0.0797, +2.4283]
  (CI excludes 0, barely); cand 03 −0.0413 [−1.2952, +1.2469]
  (null). **Both-candidates gate: FAIL.**
- Secondary window (41–60): cand 02 −1.2077 [−3.3852, +0.9168]
  (opposite sign from its own primary window); cand 03
  −2.1656 [−3.7864, −0.5599] (negative, excluding 0 — opposite of
  02's primary). Signs are inconsistent across windows AND across
  candidates.
- **L2 specificity: VIOLATED in cand 03** — random − noop =
  −2.1187 [−3.0397, −1.2581]: random editing moves the instrument
  MORE than the informed contrast does. (02: −0.93 [−2.29, +0.42],
  spans 0.)

## Adjudication

Under the frozen gates, L1 fails and L2 fails in one candidate. The
pattern — window-dependent sign flips, candidate disagreement,
non-specific response to random editing, near-zero correlation with
the stable code instrument — is the signature of an estimator
dominated by high-dimensional Gaussian-MI noise (198-dimensional
joint covariances at T ≈ 480–2,100) rather than of a physical
coupling. Registered conclusion: **the paper-as-typeset quantity
shows no reliable connection to plastic heredity at feasible sample
sizes; the isolated 02 primary-window interval is not credible
against its own secondary-window reversal and the specificity
violation.** The registered L1 lean (null) is scored a HIT; the L2
violation is reported without rescue.

## The completed instrument map (chapter 5)

| reading of the page | pipeline | heredity coupling |
|---|---|---|
| extracted (phantom denominator) | normalized, macro halves | null (Phase I) |
| typeset on macro halves | unnormalized atom identity | null (erratum adjudication) |
| typeset multivariate (paper-faithful) | this module | no coherent coupling; unstable estimator |
| authors' code (revised Φ_R) | macro halves, 9-atom sum | RESPONSIVE — prospectively confirmed (J), dose-graded in 03 (K1), carried by downward causation (K2) |

The plastic-heredity connection exists in exactly one reading of
Φ-r: the one the authors implemented, not any reading of the one
they printed.
