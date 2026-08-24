# Preregistration — C7 intervention campaign under code-faithful Φ_R

**Frozen 2026-08-17, before any campaign code was written or run.**
Append-only after this point: results and errata go below the RESULTS
marker; nothing above it may be edited.

## Background

The paper's C7 claims that selecting single-molecule interventions to
maximize Φ_r raises self-replicator persistence (874±233 vs control
716±198) and consistency (0.52 vs 0.38), while minimizing lowers them
(559±99; probability 80% vs 88%). Our Ψ-based C7 attempts (three scoring
implementations) never produced a verified bidirectional Φ manipulation
and found null SR effects. Φ_R — the quantity the authors' public code
actually computes — has never been steered in this system. The sister
replication found Φ_R responds to heredity manipulation but its
Φ-surrogate steering did not move realized Φ_R (manipulation check
failed there); this campaign is the GARD-side analog with a pointwise
window-fit scorer.

## Design

- Universe: coarse (dt=0.4, max_events_per_step=24.0), N_g=100, 100
  generations — identical to the recorded Ψ-based C7 coarse campaign.
- Arms (100 runs each, seeds 0–99 matched across arms):
  1. **control** — no intervention.
  2. **max** — after every fission (once ≥ WINDOW=100 steps of history),
     apply the single-molecule add/delete (2·N_g candidates, same
     enumeration as `interventions.py`) that maximizes pointwise Φ_R of
     the hypothetical transition (current → candidate) under the
     code-faithful macro pipeline fitted to the trailing 100-step window.
  3. **min** — same, minimizing.
  4. **random** — uniform pick from the same candidate list (edit-
     specificity control), using a dedicated RNG stream per run.
- Scorer: `phi_r_point.py` — window fit = CLR (repo convention,
  pseudocount 1.0, drop last) → dead-filter/z-score on window stats →
  lag-1 MI matrix → Fiedler halves → macro edge; pointwise 16-atom
  lattice with pointwise-MMI redundancy evaluated at the query
  transition; Φ_R = the verified nine-atom sum. **Equality gate**
  (precondition for launch): the fit/evaluate path must reproduce
  `phi_r_code_local` on the window's own consecutive pairs to ≤1e-10.
- Logged per run: replicator metrics (persistence, persistence_1k,
  probability, consistency, consistency_1k, episode_mean,
  time_to_first_pct), per-generation SR probability, realized whole-run
  Φ_R (`phi_r_code_local` mean), realized whole-run printed-Ψ mean
  (`phi_r_local` mean, cross-instrument), and the chosen edit sequence.

## Registered tests (α = 0.05; Mann–Whitney, nan-omitted)

- **T1 Manipulation-check gate**: realized Φ_R, max vs min, one-sided in
  the steered direction (max > min). ADDITIONALLY max > control and
  min < control one-sided, reported. The gate passes only if max-vs-min
  p < 0.05. **If the gate fails, no Φ_R-causal interpretation of SR
  outcomes is drawn** — the campaign is still reported in full as a
  scorer-effect study.
- **T2 Primary SR outcomes** (the paper's claim): persistence and
  probability; max > control one-sided AND min < control one-sided.
  The paper's C7 replicates only if T1 passes and both primaries pass
  in both directions.
- **T3 Secondary**: consistency_1k, episode_mean, per-generation trend
  slope — reported with two-sided tests, no gate.
- **T4 Random-arm validity**: random vs control on both primaries,
  two-sided; p > 0.05 expected. If random differs from control, edit
  specificity is compromised and T2 effects cannot be attributed to
  Φ_R-directed selection.
- **T5 Cross-instrument**: realized Ψ, max vs min (two-sided,
  descriptive) — does steering the code gauge move the printed gauge?

## Registered predictions (honest leans, two-sided reporting regardless)

- T1: uncertain. Prior attempts at one-molecule Φ steering mostly failed
  manipulation checks at fine granularity and only max-side at coarse;
  the pointwise Φ_R scorer is new. No confident lean.
- T2 (conditional on T1 passing): lean **null** — the Ψ-based C7 was
  null, and the sister system found no Φ_R→heredity control direction.
- T4: null required for validity.
- T5: lean near-zero (the two instruments are nearly uncorrelated
  observationally, r = −0.26 across runs).

## Interpretation table (frozen)

| T1 gate | T2 both primaries both directions | reading |
|---|---|---|
| pass | pass | paper's C7 replicates under the code instrument |
| pass | fail | Φ_R is steerable but SR outcomes don't follow — C7 not replicated, and the causal arrow Φ_R→SR is directly constrained |
| fail | (any) | no working Φ_R manipulation at one-molecule granularity — C7 untestable at this intervention strength; report as such |

Multiple-comparison note: T2 involves 2 outcomes × 2 directions with
all four required to pass — conjunctive, no correction needed. T3/T5
are descriptive.

---

# RESULTS (append-only below this line)

**Campaign run 2026-08-17** (`interventions_phir.py`; 400 runs in 82 s on
12 workers; equality gate 0.00e+00 at launch; smoke 3×4 passed first).
Rows: `results/interv_phir_rows.pkl`; tests:
`results/interv_phir_summary.json`.

## Arm table (mean±sd, n=100/arm)

| arm | persistence | probability | realized Φ_R | realized Ψ | edits |
|---|---|---|---|---|---|
| control | 305.9±143.0 | 0.385±0.262 | 0.957±0.200 | −1.229±0.601 | 0 |
| max | 304.7±161.8 | 0.398±0.281 | 0.907±0.187 | −1.198±0.696 | 88.0 |
| min | 289.9±158.2 | 0.382±0.272 | 0.908±0.198 | −1.250±0.578 | 88.0 |
| random | 286.3±143.2 | 0.381±0.266 | 0.915±0.181 | −1.176±0.618 | 88.0 |

(Control-arm realized Φ_R/Ψ exactly reproduce the recorded sign-regime
values from recovery_phir.json — determinism cross-check.)

## Registered tests

- **T1 manipulation check: GATE FAILS.** max vs min on realized Φ_R
  p=0.580 — the arms are statistically identical (0.907 vs 0.908).
  max is NOT above control (p=0.975; realized Φ_R actually sits slightly
  below control in every edited arm, including random); min<control
  p=0.041. ~88 opposite-signed Φ_R-directed one-molecule edits per run
  produce zero differential effect on the gauge they optimize.
- **T2 primaries: all null** (p=0.15–0.63), reported without causal
  interpretation per the gate.
- **T3 secondary** (descriptive): consistency_1k max-vs-control p=0.034
  two-sided, min-vs-control p=0.060 — both arms *lower* than control,
  i.e., a perturbation effect, not a directional Φ_R effect.
- **T4 random-arm validity: PASSES** (persistence p=0.354, probability
  p=0.731) — edits per se do not change SR outcomes.
- **T5 cross-instrument: null** (Ψ max-vs-min p=0.445).

## Adjudication (frozen table)

Row 3 fires: **manipulation-check gate FAILED — no working Φ_R
manipulation at one-molecule granularity; C7 untestable at this
intervention strength.** This now makes the manipulation-check failure
unanimous across four independent scorer implementations (three Ψ-based
+ this pointwise code-faithful Φ_R one) and two laboratories (the sister
replication's linearized-flow surrogate also failed its manipulation
check): in GARD-class chemistries, single-molecule edits chosen to steer
Φ do not move realized Φ of either definition. The paper's +22%/−22%
intervention effects would require an intervention channel that no
reconstruction has produced; the operational details of their
intervention procedure remain the decisive open question for the
authors.
