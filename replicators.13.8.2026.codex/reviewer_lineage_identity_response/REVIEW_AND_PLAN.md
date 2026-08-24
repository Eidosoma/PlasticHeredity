# Review and frozen implementation plan: lineage identity tests 2--4

**Status:** implemented protocol and runnable workflow  
**Classification:** reviewer-prompted prospective simulation conditional on a
previously observed capable-rule cohort  
**Isolation:** all new writes remain below this folder

## Scientific question

The reviewer asks whether a post-break form depends on its lineage or whether
any state under the same catalytic rule reaches the same destination. That is a
stronger question than the operational F12 endpoint. F12 certifies a sequence
of adjacent parent-to-selected-daughter similarities and usually does not
produce one episode-wide coherent composition. The primary identity object in
this package is therefore a strict coherent-eight episode; F12 is retained as
a non-rescuing control.

The study distinguishes three propositions:

1. daughters within one strict episode are more mutually identifiable than
   daughters from strict episodes in other lineages under the same rule;
2. independent futures cloned from one strict form remain closer than futures
   cloned from initially distinguishable forms in other lineages;
3. the rule supports at least two recurrent, finite-horizon stable forms from
   independent random starts.

A failure limits lineage-specific stable-identity language. It does not make a
correctly detected operational F12 sequence a software false positive.

## Input boundary and rule cohort

The only outcome-bearing selection input is
`../results/regime_confirmation/confirmation_states.csv`. A matrix is capable
when its archived REGCONF campaign contains at least one `primary_all8` event
in both candidates across the five landmarks and 128 futures. There are 134
such shared matrices. Exactly 50 are selected by sorting
`SHA256("2026082002403|matrix_id")`; archived event magnitude is not used for
ordering.

The selected beta matrices are regenerated from the sealed REGCONF beta seed
contract. New random initial compositions and dynamics use the package's
disjoint seed domain. Initial-composition seeds are shared across candidates;
dynamics seeds are candidate-specific. Existing trajectories, Fable results,
paper-final-attempt folders, NewIdeas, and manuscript prose do not supply new
outcomes.

Inference is explicitly conditional on these 50 previously capable rules. It
cannot estimate strict-event prevalence in arbitrary beta matrices.

## Simulation and endpoint contract

Each of the 50 rules is run under candidates 02 and 03 from 128 independent
random initial compositions for 256 fissions. Fissions 1--32 are burn-in. The
remaining 224 fissions form seven non-overlapping F32 windows for strict B.
The F12 control uses 18 non-overlapping 12-fission windows and leaves the final
eight fissions unused, preventing an unbounded-horizon search from silently
changing either endpoint.

Within each window, the first unrounded `H<=0.90` boundary is the break. The
strict search begins on the next fission and requires:

- eight consecutive boundaries with unrounded `H>0.90`;
- all 28 daughter pairs with unrounded `H>0.90`; and
- every daughter at inclusive `H<=0.85` from the break parent.

Strict B is the final daughter of the earliest qualifying episode in the
earliest qualifying F32 window. The F12 control is the final daughter of the
earliest post-break run of three in the earliest qualifying F12 window, without
imposing episode-wide coherence.

The first 20 qualifying lineages in seed order form each identity bank. If the
fixed 128 starts supply fewer than 20 strict episodes, a sealed extension runs
indices 128--255. Extension trajectories may fill identity banks but never
enter the attractor census. A cell still below 20 remains in every table, is
marked underfilled, and fails the adequacy component of an all-candidate gate.

## Test 2: matched sibling-versus-stranger baseline

Each episode is divided into daughters 1--4 and 5--8. The within value is the
cosine similarity between those two centroids. Cross values apply the same
axes to every ordered pair of different lineages under one rule and candidate.
This avoids comparing adjacent parent/daughter values with a differently
defined cross-lineage summary.

Per rule, report sample sizes, medians, median gap, common-language AUC
`P(within > cross)`, nearest-episode identity accuracy, empirical range
overlap, and the fraction of cross values inside the observed within range.

The corrected strong gate requires, separately in both candidates:

- all 50 banks contain 20 strict episodes;
- the 10,000-repeat whole-rule bootstrap lower 95% bound for mean rule-level
  AUC exceeds 0.75; and
- the corresponding lower bound for mean rule-level median gap exceeds 0.05.

The reviewer's literal “any overlap” criterion is reported but is not treated
as a statistically reliable primary gate.

## Test 3: independent post-break forks

Each selected B state launches two independent eight-fission futures. At every
generation the two corresponding selected daughters are compared. The sibling
score is the minimum of those eight similarities.

Stranger comparisons use every ordered different-lineage pair under the same
rule whose starting B states have `H<=0.85`. The stranger score is the maximum
corresponding-generation similarity, making “stays distinguishable” a strict
trajectory-level condition. Future outcomes never affect eligibility.
An incomplete sibling fork receives score 0; an incomplete stranger comparison
receives score 1. Extinction therefore cannot manufacture either kind of pass.

The corrected strong gate mirrors test 2: complete banks, at least one eligible
stranger pair in every rule, and whole-rule lower bounds above 0.75 for AUC and
0.05 for the median sibling-minus-stranger gap in both candidates. Literal
readouts are the fraction of sibling trajectories with minimum `H>0.90` and
stranger trajectories with maximum `H<=0.90`. A rule with no initially
distinguishable stranger is shared-destination evidence under the literal
test, not missing data.

## Test 4: finite-horizon attractor census

Only the fixed 128 random starts enter the census. Rolling post-burn-in windows
are coherent when every pair exceeds the chosen H threshold. Overlapping
windows merge into residence episodes. Their medoids are clustered by
deterministic complete linkage, which guarantees within-cluster coherence.

The primary stable-form definition requires:

- coherent residence length 8 at `H>0.90`;
- support from at least 8 independent random starts; and
- either at least 16 continuously resident fissions or departure followed by
  re-entry after at least 8 outside fissions in at least 4 starts.

Stable clusters are ordered by start support, durable support, episode count,
and cluster identifier. A deterministic greedy census retains a cluster as a
distinct form only when its medoid is `H<=0.85` from every already retained
form. A rule survives the reviewer's strong alternative-form interpretation
when at least two forms remain. Zero and one are separate reported failures.

The complete 27-cell sensitivity grid crosses residence lengths `{4,8,16}`,
start supports `{4,8,16}`, and distinctness cutoffs `{0.80,0.85,0.90}`. The
durable-start requirement is `min(4, start_support)`. Every cell is reported;
none may replace the primary definition.

## Outputs and integrity

The analysis emits rule-level and observation-level baseline tables, fork
scores and trajectories, primary and sensitivity census tables, stable-form
medoids, candidate summaries, four figures, a scientific report, appendix, and
proposed manuscript/reviewer language. Proposed language never modifies the
source manuscript.

The `prepare` stage seals source hashes, the rule list, seed domains, endpoint
inequalities, sample sizes, analysis gates, and claim boundary before new
simulation. Checkpoints record the protocol identifier and can be resumed.
Completion requires a full independent deterministic replay of every fixed,
extension, and fork checkpoint, exact discrete equality, zero boundary-H
error, verified output checksums, and a final `complete: true` audit.

## Failure and interpretation rules

- Candidate results remain separate; one contract cannot rescue the other.
- Underfilled banks fail adequacy but are still reported.
- Lack of distinguishable stranger states is evidence against lineage-specific
  identity under the literal test.
- A one-form rule fails alternative-form heredity for that rule, but does not
  erase an operational F12 event.
- F12 controls are descriptive and cannot rescue strict-B gates.
- “Stable form” means recurrence under this finite-horizon assay, not a proof of
  an infinite-time dynamical attractor.
