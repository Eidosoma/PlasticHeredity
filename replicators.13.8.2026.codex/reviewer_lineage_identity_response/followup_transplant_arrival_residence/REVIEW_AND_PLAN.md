# Frozen implementation contract: strict-B transplant and residence

## Evidence boundary

Inputs are restricted to the verified lineage-identity campaign's 50-rule
selection, strict-B bank, stable-form medoids, protocol, and verification
audit, plus the frozen simulator source.  The first three strict B rows in
bank order supply each rule/candidate cell.  Existing outcomes choose no arm,
permutation, future, or favourable analysis.

`NewIdeas` supplies hypotheses only and is excluded from scientific inputs.
The campaign is conditional on previously strict-capable rules and observed B
donors.  Candidate 02 and 03 results remain separate.

## Primary factorial

For each rule, a single molecule-label permutation is selected from 4,096
seeded proposals by minimum mean original-versus-permuted B similarity over
the six frozen donors.  Each candidate, rule, and donor launches 128 common-
seed F32 futures in four arms:

1. native B under native beta;
2. permuted B under native beta (exact abundance-spectrum-matched stranger);
3. native B under jointly permuted beta; and
4. permuted B under permuted beta (exact isomorphism control).

Where possible, the least-similar different-lineage B at initial `H<=0.85`
is an additional descriptive natural-stranger arm.  Missing natural strangers
are reported and never affect a primary gate.

The primary target is final B.  The strict episode medoid and cosine cutoffs
0.85/0.90/0.95 are non-rescuing sensitivities.  Arrival, target capture,
residence, departure/re-entry, target-independent coherence, survival, and
first-break hazard are scored through F32.  Target capture requires eight
states individually and mutually strictly above the registered threshold.

## Frozen classifications

Rule-level means are equally weighted and bootstrapped 10,000 times.

- Strong lineage identity requires native F16 capture lower 95% bound above
  0.40, native-minus-state-only point difference at least 0.20, and its lower
  bound above 0.10 in both candidates.
- Shared rule destination requires state-only F16 capture lower bound above
  0.40 and the 90% native-minus-state-only interval wholly inside +/-0.10 in
  both candidates.
- Strong transient status requires upper 95% bounds below 0.25 for native and
  state-only F32 capture in both candidates.
- Rule-conditioned rehoming requires the joint-isomorphism audit plus both
  state-only and rule-only arms favouring their rule-designated target over
  their launch target by at least 0.20 with lower bounds above 0.10.
- The isomorphism audit requires native-versus-inverse-joint 90% intervals
  within +/-0.03 for F16 capture, F32 capture, and first break by F8.

If no classification passes, the result is mixed/underdetermined.  No
sensitivity or natural-stranger cell can rescue a primary verdict.

## Rare-form challenge

Matrices 11, 54, and 63 are explicitly post-hoc selected case studies.  Each
of their two candidate-specific medoids launches intact and from eight
mass-preserving perturbations.  A pre-seal feasibility audit showed that the
original four-molecule proposal cannot reach the registered `H=0.85--0.95`
band for these concentrated forms.  The corrected frozen ladder uses the
smallest dose in `{4,8,12,16,20,24}` yielding eight starts with own-form
`H in [0.85,0.95]` and other-form `H<=0.85` among 4,096 seeded proposals per
dose.  Failure to find eight fails adequacy.

Each intact or perturbed start launches 128 F32 futures.  An individual rule
supports an exceptional two-basin interpretation only when both forms and
both candidates have perturbation-pooled own capture lower bound above 0.50,
cross capture upper bound below 0.20, and first-captured-form origin accuracy
lower bound above 0.75.  No result generalizes beyond the passing rule.

## Integrity

All checkpoints are atomic and protocol-bound.  Extinction and incomplete
futures score as no capture and remain visible.  Completion requires a full
independent replay of every future, exact discrete equality, zero H error,
verified output checksums, and a final `complete: true` audit.

