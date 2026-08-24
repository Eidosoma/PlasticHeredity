# Phase H — Strict coherent eight-fission episode occurrence replication
# PREREGISTRATION (sealed 2026-08-14, before any scientific matrix was generated)

This module is an external clean-room OCCURRENCE replication of an
operational event reported by an independent agent ("Codex") that
separately replicated the same pre-print. It is an occurrence
experiment only. This module contains NO prediction, NO molecular
intervention, NO beta surgery, NO controller selection, and NO model
fitting. Phase G5 outcomes, Codex code, Codex branch data, and Codex
fitted models were not used to design, tune, or alter anything below.
The Fable simulator contracts, matrix distribution, and parameters are
the frozen ones (`sim.py`, unchanged since the replay-gate campaign);
no engine or parameter changes.

Registered event name: **STRICT_BREAK_COHERENT8_DISTINCT**
(long form: a strict coherent, old-anchor-distinct eight-fission
hereditary episode). The registered artifacts do not call this event
"stable 8", an attractor, a compotype, memory, or regime switching.

## Scientific question

Does the Fable clean-room GARD implementation naturally generate a
break followed by a distinct, mutually coherent eight-fission
hereditary episode, under the identical frozen operational definition
used by the other clean room? The external benchmark is approximately
1.81–2.11% of futures. Nothing here is tuned to reproduce 2%; because
the simulator contracts differ between clean rooms, exact rate
agreement is descriptive, not the primary gate.

## Exact primary endpoint (frozen)

For each independent F32 future launched from a restored post-fission
state (boundaries indexed t = 1, 2, …, k with k ≤ 32 realized
boundaries; H(t) = cosine(parent_t, selected_daughter_t), unrounded
float64; candidate-specific selected-daughter semantics exactly as in
`sim.run_fissions`):

1. First break: the smallest b with H(b) ≤ 0.9. The old anchor P_old
   is the full pre-fission PARENT composition at boundary b (not the
   breaking daughter, not the restored landmark state, not a
   centroid). A no-break future is negative.
2. Strictly after b, search for eight consecutive inherited
   boundaries: H(t) > 0.9 for t = r, …, r+7 with r ≥ b+1. The run
   need not begin immediately after the break; failed intermediate
   runs may reset; a later qualifying run may count; an eight-run
   entirely before the first break does not count. All windows of
   eight consecutive inherited boundaries with r ≥ b+1 and
   r+7 ≤ min(k, 32) are enumerated in increasing r (overlapping
   windows inside longer runs included); each is an ELIGIBLE window.
3. Let D_1…D_8 be the selected daughter compositions at the window's
   eight boundaries.
4. Episode-wide coherence: H(D_i, D_j) > 0.9 for all 28 pairs i < j.
5. Old-anchor distinctness: H(D_i, P_old) ≤ 0.85 for i = 1…8.
6. Both the break and the certification boundary (r+7) must lie
   within the 32-fission horizon.
7. Semantics: H > 0.9 inherited; H ≤ 0.9 break (H = 0.9 is a break);
   pairwise coherence strictly > 0.9 (= 0.9 fails); distinctness
   inclusively ≤ 0.85 (= 0.85 passes); unrounded float64 throughout;
   an event certified before later extinction remains positive;
   extinction before certification is negative; the FIRST qualifying
   window is recorded as the primary event window (all eligible
   windows are persisted).

The future is POSITIVE iff at least one eligible window passes both
coherence and distinctness.

## Cohort (fully prospective; frozen before generation)

- Seed architecture: entropy = sha256("replication-strict8-domain-
  2026-08-14") via `cohort.domain_entropy("strict8", "2026-08-14")`;
  spawn domain 25 (next unused). Keys: (25,0,m) catalytic matrix;
  (25,1,m) initial state; (25,2,cand_i,m) natural main trajectory;
  (25,3,cand_i,m,lm,b) branch future b at landmark lm.
- 200 completely fresh catalytic matrices (m = 0…199), shared across
  frozen candidates 02 and 03; frozen matrix distribution
  (`sim.make_beta` defaults); no parameter or engine changes.
- Natural main trajectories only (100 fissions, the frozen
  contract); no controller-written states. Restored post-fission
  landmarks at fissions 20, 35, 50, 65, 80. Main-path
  extinction follows the frozen Fable contract: a landmark beyond the
  trajectory's realized fissions is dropped, never retried; realized
  state counts are disclosed in the results (nominal 2 × 200 × 5 =
  2,000 restored states).
- 128 independent F32 futures per state; branch futures never
  retried. Prospective halves by branch index: half A = 0–63,
  half B = 64–127. Nominal total 256,000 futures.
- A second complete deterministic replay of all futures is run and
  compared by SHA-256 over the float64 H traces and classification
  records (exact-replay gate).
- No state is selected or filtered using v2 risk, R_Q, atlas
  similarity, inheritance history, Phase G outputs, or observed
  strict-event outcomes.

## Persistence (frozen)

For EVERY future: branch seed key (deterministic recipe above), the
H trace (archived float32; classification and replay hashing use the
original float64), first-break boundary, all eligible windows with
their continuous components, event classification, and component
outcomes. Full composition vectors are retained for: all restored
landmark states; P_old + the eight episode daughters of every
POSITIVE future's primary window; and a registered outcome-blind
audit sample — every future whose flat index
((cand_i·200 + m)·5 + landmark_index)·128 + b ≡ 0 (mod 997), for
which parents and daughters of the full future are stored to allow
independent recomputation of H. Everything else is exactly
reconstructable from the seed recipe.

## Registered component decomposition (secondary; cannot replace the primary endpoint)

Per future: (1) first break within F32; (2) any post-break eight-run;
(3) all-pair coherence of the FIRST eligible window; (4) old-anchor
distinctness of the FIRST eligible window; (5) full strict endpoint;
(6) min of the 28 pair similarities (first eligible window);
(7) max similarity of any episode daughter to P_old (first eligible
window); (8) coherence margin = min_pair − 0.9; (9) distinctness
margin = 0.85 − max_anchor; (10) full consecutive inherited run
length containing the primary window, where observable.

## Primary analysis (frozen)

Candidates and branch halves are never pooled. Four primary cells:
02/A, 02/B, 03/A, 03/B. Per cell: branch-pooled occurrence rate;
95% whole-matrix bootstrap interval (4,096 draws, fixed seed 41; the
catalytic matrix is the bootstrap unit and carries all its states and
branches per draw); total events; number and fraction of matrices
with ≥ 1 event; number and fraction of restored states with ≥ 1
event; median and maximum events contributed by one matrix; event
timing (first break, run start, certification: medians and ranges
over positives); exact-replay equality and artifact hashes.
Descriptive additions (labeled descriptive): equal-state macro rate;
equal-matrix macro rate; branch-half per-state probability agreement
(noisy at 64 branches for a rare event).

## Primary replication gate (frozen)

Independently replicated iff: (1) strict-event rate > 0 in all four
cells; (2) whole-matrix bootstrap 95% lower bound > 0 in all four
cells; (3) every scientific future replays exactly; (4) no candidate
pooling, replacement matrices, endpoint revision, or post-outcome
threshold changes. Breadth of matrix participation is reported
prominently; a broad-distribution claim requires events across many
matrices, not merely many branches.

## External benchmark comparison (applied only after sealing the Fable result)

External cells (occurrence rates): 02/A 0.01869; 02/B 0.01809;
03/A 0.02089; 03/B 0.02109 (1,158–1,350 events, 119–143 event-bearing
matrices per cell). Not fitting targets, not pass/fail thresholds.
Registered conclusion rule:
- **A (phenomenon and rate numerically compatible):** gate passes AND
  every cell's 95% whole-matrix CI contains the corresponding
  external cell rate.
- **B (phenomenon replicated, rate contract-sensitive):** gate passes
  but at least one cell's CI excludes its external rate. Because the
  simulator candidates differ between clean rooms, outcome B is
  legitimate and is not a failure.
- **C (partial or failed replication):** one or more cells fail the
  frozen occurrence gate.

## Sealing and smoke policy

Sealed together, before any scientific matrix is generated: this
preregistration; `run_strict8_occurrence_replication.py` (endpoint +
cohort + analysis, one file); `test_strict8_endpoint.py` (the
registered fixture list below); the seed tag; the branch-half
assignment; SHA-256 hashes of these files plus `sim.py` and
`cohort.py`, recorded in `results_strict8_occurrence/SEAL.json`.
A tiny smoke run (separate seed tag "strict8-smoke-2026-08-14",
2 matrices, 8 branches) may exercise I/O and replay ONLY; it prints
and stores no event counts or rates, and the protocol is not altered
after it.

## Registered endpoint fixtures (all must pass before sealing)

1 no-break → negative; 2 break + seven inherited → negative; 3 break
+ eight inherited but one pair ≤ 0.9 → negative; 4 coherent eight-run
with one daughter > 0.85 to P_old → negative; 5 valid break +
coherent/distinct eight-run → positive; 6 pre-break eight-run →
negative unless a new qualifying post-break run occurs; 7 interrupted
post-break run then a later valid run → positive; 8 certification
exactly at 32 → positive; 9 certification after 32 → negative;
10 exact thresholds (H = 0.9 non-inherited; pair 0.9 fails; anchor
0.85 passes); 11 molecule-label permutation invariance; 12 candidate-
specific selected-daughter semantics; 13 exact replay from branch
seed; 14 positive-before-extinction and extinction-before-
certification.

## Claim boundary

A passing result supports only: "An independent Fable clean-room GARD
implementation prospectively generates rare break-followed-by-
distinct-coherent eight-fission hereditary episodes under the frozen
operational definition." It does not establish an attractor,
recurrence, autonomous return after perturbation, permanent
stability, a compotype, biological memory, individuality, prediction
from the pre-event state, causal control, or real prebiotic
chemistry. No prediction or intervention is tested in this module.
Restored states, branch seeds, first-break states, eligible windows,
strict-event windows, and component outcomes are frozen and archived
for a later, separately registered study.
