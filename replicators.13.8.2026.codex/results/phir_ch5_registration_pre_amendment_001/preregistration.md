# Codex Chapter 5 Φ-r / plastic-heredity preregistration

## Registration scope

This document prospectively fixes an independent Codex study of how two
published formulations called Φ-r relate to the already established
plastic-heredity process in the Codex GARD reconstruction. It is a white-room
implementation: no Fable code, matrices, states, seeds, fitted objects,
selected interventions, result tables, or controller objects may enter the
program. Public typeset material and the public PhiRL repository are source
specifications, not result targets.

The program has two disjoint scientific cohorts:

- a 24-catalytic-matrix pilot, which is inspected before any larger run; and
- a 48-catalytic-matrix prospective confirmation, which requires a separate
  user-created authorization artifact after pilot review.

The pilot and confirmation are never pooled. Completion of the pilot cannot
automatically launch the confirmation. All scientific runs execute detached,
checkpoint by whole catalytic matrix, and undergo a complete second replay.

## Scientific questions

1. Does a revised Φ-r reading rise and fall with frozen interventions that
   causally stabilize or destabilize heredity?
2. Does revised Φ-r read whether a lineage is currently inside a short
   hereditary episode?
3. Does either Φ-r formulation predict future `JOINT_BREAK_RUN3` risk beyond
   the already frozen process-risk predictor and direct history?
4. Is any response graded across a frozen intervention dose ladder, and which
   of the 16 ΦID atoms carries it?
5. Can a bounded Φ-directed edit search move the Φ gauge, and if so does
   hereditary behavior follow?
6. Do results depend on measuring molecular-update time versus fission time?

These questions distinguish a responsive gauge, a state marker, a predictor,
and a causal controller. None is treated as synonymous with the others.

## Frozen simulator and target

The existing Codex simulator is not altered. Candidate 02 and candidate 03
retain the contracts already used for the JOINT_BREAK_RUN3 and intervention
campaigns. They are never pooled to rescue disagreement.

Strict inheritance is unrounded cosine similarity `H > 0.9` between a
pre-fission parent and its selected continuing daughter. `JOINT_BREAK_RUN3`
is positive within F12 only when a strict break (`H <= 0.9`) is followed,
strictly later, by three consecutive inherited fissions. Extinction before
certification is negative; certification before later extinction remains
positive.

The frozen process-risk predictor is copied from
`results/scaled5/frozen_models.npz`, SHA-256
`9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af`.
It is not refit, recalibrated, simplified, or selected using Chapter 5
outcomes.

## Information instruments

All instruments use float64 and nats. Molecular counts receive additive 0.5
zero replacement, closure, CLR transformation, and removal of the final CLR
coordinate. Coordinates with standard deviation no greater than `1e-8` in
the past-only window are excluded. Remaining coordinates are standardized
using that window only.

A symmetric lag-one Gaussian mutual-information graph is built from the
standardized series. A `1e-6` graph floor makes it connected, and the
unnormalized Laplacian Fiedler vector gives a deterministic bipartition.

The registered readings are kept separate:

1. **Typeset Φ-r**: the unnormalized, multivariate
   `I(X_t;X_t+1) - Σ_i I(X_t^i;X_t+1)` quantity visually verified from the
   public typeset page.
2. **Revised Φ-r**: the nine-atom ΦID sum reconstructed independently from
   public PhiRL commit
   `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`, after averaging each Fiedler
   half into one macro-variable.
3. **Text-extraction ratio**: typeset numerator divided by whole-system MI,
   retained only as a registered negative/control reading.
4. **Unbundled ΦID**: all 16 two-source atoms, plus causation, emergence, and
   synergy-persistence summaries.

The public PhiRL port must match a fresh synthetic parity fixture to `1e-12`
before registration. The public repository is not vendored into this tree.

## Clocks and windows

- Primary molecular clock: the last 512 observed compositions, including
  growth updates, fission jumps, and intervention jumps.
- Growth-only sensitivity: the same rolling buffer but only growth-to-growth
  transition pairs are fitted and scored.
- Generational clock: the last 20 continuing-daughter compositions.

All fits are prefix-only. The primary instrument deliberately includes
fission and intervention transitions because those are part of the measured
assembly process; the growth-only sensitivity reveals whether conclusions
depend on that choice. Raw molecular trajectories are never persisted.

## Scientific cohort shared structure

For each catalytic matrix and each candidate, two natural replicates run for
100 fissions. Natural states are not selected by risk or outcomes. Scores are
recorded from generation 20. Restored landmarks 20, 35, 50, 65, and 80 each
receive 64 F12 futures. Fixed halves are branches 0–31 and 32–63. The natural
generation-60 state launches all intervention programs.

The pilot uses 24 matrices. Confirmation uses 48 entirely new matrices from
a separate matrix seed domain. Each phase receives independent purpose-keyed
streams for matrices, initial states, main paths, F12 futures, controller
actions, dose futures, probe selection, probe screening, probe confirmation,
bootstrap, randomization, and replay.

Arms within a paired experiment start future simulation from the same stream
key; arm identity never appears in that future key. These are common random
streams, not necessarily identical realized futures after states diverge.

## Experiment A — six-arm causal bridge

At each fission after the natural generation-60 launch, apply one of:

- `MODEL_STABILIZE`: exhaustive legal substitution with the smallest frozen
  JOINT_BREAK_RUN3 prediction;
- `MODEL_DESTABILIZE`: exhaustive legal substitution with the largest frozen
  prediction;
- `RULE_STABILIZE`: the frozen outgoing-influence physical rule-down edit;
- `RULE_DESTABILIZE`: the corresponding rule-up edit;
- `RANDOM`: one uniform legal substitution from an independent action stream;
- `NOOP`: no substitution.

Every arm runs 60 fissions. The primary analysis averages fissions 31–60.
The principal contrasts are stabilizing minus destabilizing for inherited
fraction and each information reading. Candidate and replicate are reported
as four separate cells. A positive confirmatory response requires positive
effect, positive 95% whole-matrix bootstrap lower bound, and Holm-adjusted
one-sided whole-matrix sign-randomization `p < 0.05` in all four cells.

No-op execution through the traced callback must be bitwise identical to the
plain simulator. Random-minus-no-op hereditary probability is tested for
equivalence within ±0.025.

## Experiment B — current hereditary-state reading

A state is `SR_RUN5` when its trailing inherited run is at least five
fissions. Within each eligible lineage, compute mean reading during SR_RUN5
minus mean reading outside SR_RUN5. Lineages without both conditions do not
contribute a contrast; their counts remain reported. Lineage contrasts are
averaged only within whole matrix. Natural and controlled trajectories are
reported separately.

The revised-Φ primary state-reading gate requires a positive controlled-state
contrast, positive 95% matrix-bootstrap lower bound, and Holm-adjusted
one-sided matrix randomization `p < 0.05` in both candidates. Replicate
sensitivity remains machine-readable.

## Experiment C — prospective F12 foresight

For every natural landmark, estimate future JOINT_BREAK_RUN3 probability in
each fixed 32-branch half. Report within-matrix-centered Spearman association
for:

- the frozen process-risk predictor (validity control);
- revised, typeset, and text-ratio Φ-r;
- causation, emergence, and synergy-persistence;
- growth-only revised Φ-r; and
- generational revised and typeset readings.

The frozen predictor validity gate requires positive centered correlation and
a positive 95% whole-matrix bootstrap lower bound in all four
candidate-by-half cells. A Φ association is called equivalent to small only
when its 90% matrix-bootstrap interval is wholly inside ±0.10; crossing zero
alone is not equivalence.

A fixed two-fold matrix cross-fit compares candidate-specific logistic
students using frozen-predictor logit plus the nine direct-history variables
against the same baseline plus the registered Φ readings. Scaling, imputation,
and fitting use training matrices only; `C=1`, `lbfgs`, and no hyperparameter
search are fixed. Positive gain means lower ordinary branch-level log loss.
Equivalence requires the complete 90% CI inside ±0.005 nats per branch.

## Experiment D — frozen edit-dose ladder

At generation 60, exhaustively score all legal one-molecule substitutions by
the frozen predictor. Select deterministic empirical quantiles 0, 20, 40, 60,
80, and 100 percent. Apply each edit once, then run 24 fissions on a fresh
common-random-stream domain. Realized outcomes never select a dose.

Within each state, correlate the stabilizing direction (negative predicted
risk shift) with the final information reading. Candidate and replicate are
kept separate. The revised-Φ dose gate requires positive mean within-state
Spearman, positive 95% matrix-bootstrap lower bound, and Holm-adjusted
one-sided randomization `p < 0.05` in all four cells. Atom and generational
results are registered secondary decompositions.

## Experiment E — bounded Φ-directed probe

At generation 60 create exactly 64 legal candidate edits when available:
model extrema, physical-rule extrema, and deterministic uniform random legal
edits with duplicates removed. Each edit receives four six-fission screening
rollouts on a screening-only seed domain. Select the largest and smallest mean
revised-Φ endpoints, with deterministic edit-order tie handling.

On a fresh confirmation stream, run `PHI_UP`, `PHI_DOWN`, one independently
random legal edit, and `NOOP` for 24 fissions. The gauge-movement gate requires
positive PHI_UP-minus-PHI_DOWN revised Φ-r with positive 95% lower bound and
Holm-adjusted `p < 0.05` in all four cells. Heredity is not assumed to follow;
equivalence requires the full 90% CI inside ±0.025 for inherited fraction.
This bounded probe is not an unrestricted controller search.

## Inference, multiplicity, and replay

The catalytic matrix is the inference unit. Repeated states, replicates,
arms, branch halves, and time points from one matrix remain together in every
resample. Major analyses use 4,096 whole-matrix bootstrap draws and 4,096
paired whole-matrix sign randomizations. Holm correction applies across the
four registered candidate/replicate or candidate/half cells within a metric
family.

Every scientific matrix is generated twice from scratch in independent
checkpoint trees. Compact states, actions, endpoints, process outcomes, Φ
scores, and RNG-dependent scientific digests must agree exactly. No failed,
extinct, or adverse scientific matrix is replaced after seed sealing. Natural
main paths use only the pre-existing bounded retry contract; intervention
futures are never retried.

## Phase advancement and stop rules

1. The 34-check validation suite must pass.
2. Registration seals protocol, source hashes, model hash, seeds, tests, and
   validation before any scientific matrix.
3. A non-scientific smoke test may exercise I/O and replay but may not report
   effect sizes, arm ordering, rates, or candidate differences.
4. Launch the 24-matrix pilot detached and stop after its exact replay,
   analysis, and reports.
5. Show the pilot results to the user. Do not create confirmation seeds or
   work until the user explicitly authorizes the exact sealed 48-matrix run.
6. Confirmation, if authorized, runs detached and remains separate.

No post-outcome changes are allowed to instruments, windows, endpoints,
cohorts, arms, edit selection, seed domains, inference, equivalence margins,
or stop rules. A failed result remains at full prominence.

## Claim boundary

A confirmed response may show that a specified information-statistical gauge
tracks or responds to controlled hereditary organization in these Codex GARD
contracts. A current-state contrast does not imply foresight. A movable gauge
does not imply causal control of heredity. No result here supports
consciousness, life, autonomous agency, biological memory, error correction,
real prebiotic chemistry, a universal origin-of-life mechanism, Φ or ΦID as
an intervention target in nature, or a literal Platonic-space/Ruliad portal.

