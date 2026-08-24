# Composite-versus-history capacity-matching review report

**Date:** 2026-08-18  
**Status:** Analysis and revision recommendation only. No change to the preprint has been made.  
**Scope:** The frozen F12 `FULL_STATE_GRAPH_HISTORY` predictor versus the nine-variable `DIRECT_HISTORY_PHASE` comparator.

## Executive finding

The reviewer identified a real methods-reporting omission and a legitimate analysis gap, but the retained data favour the preprint's predictive claim.

1. The direct-history comparator is the same development-fitted, standardized L2 logistic-ridge model with `C=0.1` as the composite. This is explicit in the preregistration and code but not stated clearly in the manuscript.
2. Equal model class and `C` do not capacity-match the comparison: the direct model has nine fitted inputs, whereas the composite has twelve development-fitted PCA scores plus the same nine inputs, for 21 fitted inputs.
3. A reviewer-prompted, no-refit rescore is possible with the retained independent-test-1 generative-null data. When the 195-coordinate block is recomputed for the same natural states using the already-generated coupling derangement, the aligned composite's `+0.0243` to `+0.0256` nat advantage becomes `-0.0114` to `-0.0208` nats. All four descriptive 95% whole-matrix bootstrap intervals are below zero.
4. A complementary exact-marginal block derangement also destroys the gain at both the retained 40-matrix and 200-matrix scales. Every nonzero cyclic whole-matrix pairing tested is negative.
5. These results disfavor a generic “more fitted coordinates/PCA capacity” explanation. They show that the 195-coordinate block must be correctly aligned with the natural state and matrix for this frozen predictor to improve on history.
6. They do **not** identify which aligned physical information matters. The signal may still be a regularized combination of composition, phase, history-correlated state, matrix propensity, and catalytic context rather than a uniquely isolated network-state interaction.
7. The exact coupling rescore presently covers one independent implementation. The originating L53/L54 model and raw state artifacts are not available in this checkout, so the originating `0.041-0.052` nat comparison has not itself received the same rescore.

The net assessment is positive for the preprint, provided the new analysis is made reproducible, labelled post hoc, and described with the remaining attribution and implementation-scope limitations.

## Reviewer comment, verbatim

> The capacity-matching concern in the composite-vs-history comparison — your other flagged analysis item — is not yet addressed. Two parts. First, the direct-history comparator's model class and hyperparameters are never stated in the manuscript (presumably they're in the preregistrations, but the methods section should say whether it's the same ridge with C = 0.1 on the nine variables). Second, the composite has 21 fitted features against the comparator's 9, plus a development-fitted PCA, so the 0.03–0.05-nat gain is open to a "more regularised features of correlated signals" explanation. Appendix D's no-PCA decompositions bear on attribution but changed the representation and fitting sequence, so they don't settle capacity. The right control is the predictive analogue of your coupling derangement: recompute the 195-coordinate block for the same natural states under a deranged matrix–state pairing, run the identical frozen pipeline (same dimensionality, approximately same marginal feature distributions, broken alignment), and score against the same already-observed outcomes. If the gain over history vanishes, the content matters and not the capacity. This needs rescoring, not new futures — the most expensive of the four open items, but still cheap relative to any campaign in the paper.

## 1. Audit of the current preprint

### 1.1 What the methods currently say

Under **Frozen F12 predictor and comparators**, the current manuscript says that the originating algorithm combines:

- nine history/phase variables;
- a 195-coordinate state/network representation;
- development-only scaling and 12-component PCA; and
- ridge logistic regression with `C=0.1` on the twelve components plus the nine direct variables.

The manuscript then enumerates the nine direct-history variables, but it does not explicitly say how the direct-only comparator is fitted. A reader cannot tell from the manuscript alone whether direct history uses:

- the same L2 logistic model;
- the same `C=0.1` regularization value;
- the same scaling and imputation contract;
- the same development cohort; or
- any hyperparameter search.

Relevant manuscript source:

- [`../PRE_PRINT_PAPER_DRAFT.md`, “Frozen F12 predictor and comparators”](../PRE_PRINT_PAPER_DRAFT.md#frozen-f12-predictor-and-comparators)
- [Current typeset PDF](../output/pdf/plastic-heredity-biorxiv-v1.pdf)

### 1.2 What Appendix D currently establishes

Appendix D correctly says that later no-PCA/offset-ridge decompositions changed both the representation and fitting sequence. Those decompositions constrain attribution to composition, static beta, and beta-conditioned state, but they are not a capacity-matched null for the original frozen 21-input algorithm.

The appendix therefore currently leaves a “regularisation-and-encoding explanation” open. That is scientifically cautious, but it also confirms the reviewer's point: the existing appendix does not test whether the original gain requires correctly aligned content from the 195-coordinate block.

Relevant source:

- [`../PRE_PRINT_PAPER_DRAFT.md`, Appendix D](../PRE_PRINT_PAPER_DRAFT.md#appendix-d-mechanistic-and-predictive-nulls-that-bound-the-claim)

## 2. What the preregistration and code actually specify

### 2.1 Originating implementation

The L53 preregistration specifies:

- `DIRECT_HISTORY_PHASE` and `FULL_STATE_GRAPH_HISTORY` in the same registered model family;
- nine named direct-history variables;
- the exact 195-coordinate graph signature;
- 12 development-only PCA components;
- `logisticRidgeC: 0.1`; and
- no hyperparameter search.

Source:

- [`s19_l53_past_observable_regime_capacity_proxy.yaml`](../original.1.8.2026.eidosoma-ai-scientist.code/arrival-of-self-replicators-eidosoma-groups-42/configs/e01/s19_l53_past_observable_regime_capacity_proxy.yaml)

The L53 runner sets:

```python
PCA_COMPONENTS = 12
RIDGE_C = 0.1
```

It constructs the direct feature matrix from the nine history columns. For the full model, it scales the 195-coordinate block on development matrices, fits development-only PCA, and concatenates the twelve PCA scores with the same nine history variables. Every non-prior model is then passed through the same `fit_binomial_ridge(..., c=RIDGE_C)` routine.

Sources:

- [`run_s19_l53_regime_capacity_proxy.py`](../original.1.8.2026.eidosoma-ai-scientist.code/arrival-of-self-replicators-eidosoma-groups-42/scripts/e01/run_s19_l53_regime_capacity_proxy.py)
- [`heredity_phi_incremental.py`](../original.1.8.2026.eidosoma-ai-scientist.code/arrival-of-self-replicators-eidosoma-groups-42/src/e01_onset_discovery/heredity_phi_incremental.py)

The shared fitting routine uses:

- median imputation with missingness indicators;
- standardization;
- scikit-learn `LogisticRegression`;
- L2 penalty by default;
- `lbfgs`;
- `C=0.1` as supplied by L53; and
- binomial success/failure sample weights.

The L53 runner checks that the nine direct variables are finite before fitting, so no extra missingness-indicator columns are expected for that comparator. Its effective fitted input count is nine. The full model's fitted input count is 21 after PCA.

### 2.2 Clean-room implementation

The separate clean-room model implementation states the same contract directly:

- direct: standardized ridge logistic regression on nine variables;
- full: standardized 195-coordinate block, PCA-12, the same scaled nine variables, and ridge logistic regression;
- `C_RIDGE = 0.1`; and
- no confirmation refit.

Source:

- [`replication/models.py`](../replicators.13.8.2026.fable/replication/models.py)

### 2.3 Conclusion of the model audit

The answer to the reviewer's first question is therefore **yes**: the direct-history comparator is the same model class with the same `C=0.1` ridge setting, fitted on the nine direct variables. The manuscript should state this explicitly.

That answer does not remove the second concern. The two models still differ in fitted input dimension and in access to a development-fitted PCA representation.

## 3. Why the capacity concern is legitimate

The registered comparison is an algorithm comparison, not a parameter-count-matched comparison:

| Model | Fitted predictor inputs | Development-fitted transforms | Final estimator |
|---|---:|---|---|
| Direct history | 9 | Imputation/standardization | L2 logistic ridge, `C=0.1` |
| Composite | 21 | State-block scaling, PCA-12, then model scaling | L2 logistic ridge, `C=0.1` |

Using the same `C` does not make these representations equally expressive. The additional components could:

- carry genuine aligned information about current state and catalytic context;
- act as correlated proxies for history or phase;
- alter shrinkage geometry and the fitted history coefficients; or
- improve prediction simply because a richer regularized representation is available.

The later decomposition models do not isolate this issue because they use different blocks, different penalties, no PCA, and sequential offset fitting. A valid capacity diagnostic should keep the original frozen 21-input pipeline intact while breaking only the alignment between the 195-coordinate block and the observed natural outcome.

## 4. Retained data relevant to the proposed control

### 4.1 Originating L53/L54 workflow

This checkout retains:

- the L53 and L54 preregistrations;
- executable runner code; and
- markdown result reports.

It does **not** retain the originating frozen model archive, transformed state table, raw confirmation states, or L54 outcome parquet files needed for an immediate exact rescore. The original runner also expects artifact paths outside this checkout. Repeating the exact originating analysis therefore requires recovering the original artifact archive or reconstructing all upstream dependencies.

### 4.2 Independent test 1: retained 40- and 200-matrix cohorts

The clean-room replication retains complete state-feature, history-feature, outcome, and frozen-model arrays at two useful scales:

- [`results/full/analysis_arrays.npz`](results/full/analysis_arrays.npz): 40 confirmation matrices, 400 states, 25,600 futures;
- [`results/full/frozen_models.npz`](results/full/frozen_models.npz);
- [`results/scaled5/analysis_arrays.npz`](results/scaled5/analysis_arrays.npz): 200 confirmation matrices, 2,000 states, 128,000 futures; and
- [`results/scaled5/frozen_models.npz`](results/scaled5/frozen_models.npz).

The 200-matrix frozen model used below has SHA-256:

```text
9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af
```

### 4.3 Independent test 1: retained GN1 natural and coupling-deranged cohort

The generative-null result retains:

- 96 source beta matrices;
- one fixed-point-free coupling permutation per matrix;
- 3,840 launch compositions across four mechanisms, two candidates, and five landmarks;
- 960 natural launch states;
- natural and null outcome probabilities in two fixed 32-branch halves;
- exact state identifiers;
- the archived frozen direct and full predictions; and
- 61,440 already-observed natural F12 futures.

Relevant artifacts:

- [`results/generative_null_decomposition/state_and_matrix_arrays.npz`](results/generative_null_decomposition/state_and_matrix_arrays.npz)
- [`results/generative_null_decomposition/state_probabilities.csv.gz`](results/generative_null_decomposition/state_probabilities.csv.gz)
- [`results/generative_null_decomposition/SCIENTIFIC_REPORT.md`](results/generative_null_decomposition/SCIENTIFIC_REPORT.md)
- [`results/generative_null_registration/frozen_full_predictor.npz`](results/generative_null_registration/frozen_full_predictor.npz)
- [`plastic_heredity/generative_nulls.py`](plastic_heredity/generative_nulls.py)

The coupling permutation was generated before GN1 outcomes from its own sealed seed domain. It is a Sattolo cycle, so every molecular label moves. Applying the same permutation simultaneously to beta rows and columns preserves the catalytic weight multiset and all matrix invariants up to relabelling while breaking the evolved label alignment with the unchanged composition.

### 4.4 Independent test 2

Independent test 2 retains frozen model bundles and compacted confirmation outcomes at the 40-, 200-, and 1,000-matrix scales. Its stored confirmation table no longer contains the raw `X9` and `X195` fields, but [`replication/cohort.py`](../replicators.13.8.2026.fable/replication/cohort.py) contains a deterministic `conf_features_unit` that reconstructs trajectory and landmark features without regenerating branch futures. A second-implementation rescore is therefore feasible but has not been run in this review.

## 5. Primary diagnostic: recomputed coupling-deranged 195-coordinate block

### 5.1 Question

Does the archived 21-input composite still outperform the same direct-history comparator when the model retains its full dimension and coefficients but the 195-coordinate state/network block is recomputed under the wrong molecular-label coupling?

### 5.2 Design

The diagnostic used only GN1 `NATURAL_GARD` states and their already-observed natural outcomes:

- 96 catalytic matrices;
- two candidates;
- five landmarks per matrix;
- 960 natural states in total;
- 64 existing F12 outcomes per state, split into fixed halves A and B;
- 61,440 scored futures;
- no new future simulation;
- no model refit;
- no PCA refit;
- no recalibration; and
- no change to the direct-history prediction.

For natural state composition `x_mg`, source matrix `beta_m`, and GN1's pre-existing fixed-point-free permutation `pi_m`, the control recomputed:

```text
phi_deranged(m,g) = state_graph_195(x_mg, beta_m[pi_m, pi_m])
```

The resulting 195-vector was passed through the exact frozen development scaler and PCA. Its twelve PCA scores replaced the aligned scores in the unchanged 21-input logistic model, while the state's original nine history variables were held fixed. The prediction was scored against the same already-observed natural F12 outcomes and the same frozen direct-history comparator.

The retained GN1 result bundle does not store the nine raw history variables, but the frozen final model is linear after PCA and scaling. The control therefore recovered the fixed history contribution from the archived aligned natural logit and replaced only the state-block logit contribution. Reconstructing the aligned prediction this way had maximum absolute error:

```text
1.1102230246251565e-16
```

This validates the algebraic substitution to floating-point precision. A permanent implementation should nevertheless reconstruct or explicitly retain the nine history inputs as an additional readback check.

### 5.3 Descriptive inference

The diagnostic used 4,096 whole-matrix bootstrap draws with temporary analysis seed `20260818`. Candidate and branch half remained separate. These intervals are reviewer-prompted, post-hoc descriptive intervals; they are not part of the sealed GN1 preregistration and are not multiplicity-adjusted.

Positive gain means lower branch log loss than the same frozen direct-history comparator. Negative gain means the 21-input model is worse than direct history.

| Candidate | Half | Aligned full gain | Aligned 95% CI | Coupling-deranged-block gain | Deranged 95% CI |
|---|---|---:|---:|---:|---:|
| 02 | A | +0.025581 | [+0.01794, +0.03330] | -0.011373 | [-0.02182, -0.00084] |
| 02 | B | +0.024272 | [+0.01637, +0.03230] | -0.012844 | [-0.02358, -0.00256] |
| 03 | A | +0.025480 | [+0.01635, +0.03463] | -0.019540 | [-0.03176, -0.00665] |
| 03 | B | +0.024821 | [+0.01578, +0.03359] | -0.020814 | [-0.03439, -0.00741] |

The aligned point estimates reproduce the sealed GN1 report's `+0.0243` to `+0.0256` range. Minor interval differences from the sealed report reflect the temporary bootstrap stream used for this diagnostic.

The aligned-minus-deranged gain differences are:

| Candidate | Half | Aligned gain minus deranged gain |
|---|---|---:|
| 02 | A | +0.036953 |
| 02 | B | +0.037116 |
| 03 | A | +0.045020 |
| 03 | B | +0.045635 |

The direct-history advantage does not merely approach zero after derangement. The wrong 195-coordinate content makes the frozen 21-input predictor significantly worse in every descriptive cell.

### 5.4 Development-mean state-block diagnostic

As an additional diagnostic, the state-block contribution was set to its development-centred mean while retaining the frozen full model's history contribution. The gain over the independently fitted direct comparator was also negative in all four cells:

| Candidate | Half | Development-mean-block gain | 95% CI |
|---|---|---:|---:|
| 02 | A | -0.020996 | [-0.03842, -0.00002] |
| 02 | B | -0.027141 | [-0.04435, -0.00790] |
| 03 | A | -0.028238 | [-0.04359, -0.00991] |
| 03 | B | -0.025351 | [-0.03917, -0.00858] |

This is not a standalone capacity-matched test because the full model's history coefficients were learned jointly with aligned state components. It is consistent with the main result: the full model needs its fitted state-block contribution at prediction time.

## 6. Distribution-shift audit

The coupling derangement preserves matrix-level beta invariants, but it need not preserve every composition-conditioned feature marginal. The control therefore requires an explicit distribution-shift audit.

### 6.1 Extreme development-envelope checks

For each candidate, a row was flagged when any of its 195 features lay more than five development-standardization units from the frozen training mean.

| Candidate | Rows | Aligned rows outside any ±5-SD coordinate | Deranged rows outside any ±5-SD coordinate |
|---|---:|---:|---:|
| 02 | 480 | 24 | 12 |
| 03 | 480 | 15 | 11 |

The loss of predictive gain is therefore not explained by an increase in the count of extreme five-SD rows. The deranged block actually had fewer such rows.

### 6.2 Marginal shifts that remain

Some composition-conditioned coordinates do shift substantially:

| Candidate | Median absolute feature-mean SMD | 90th-percentile absolute SMD | Maximum absolute SMD | L2 shift in mean 12-PC score |
|---|---:|---:|---:|---:|
| 02 | ~0.000 | 1.362 | 2.779 | 5.386 |
| 03 | ~0.000 | 1.351 | 2.760 | 5.562 |

The near-zero median occurs because many coordinates are composition-only or global-beta summaries that are invariant under the operation. The upper tail shows that the state-conditioned portion changes, as intended. Mean predictions move from `0.3943` to `0.4668` in candidate 02 and from `0.4198` to `0.5127` in candidate 03.

This is why the coupling rescore should be paired with an exact-marginal sensitivity rather than being presented as if all derived feature marginals were identical.

## 7. Exact-marginal whole-matrix block sensitivity

### 7.1 Purpose and construction

The exact-marginal sensitivity uses the retained 40- and 200-matrix independent-test-1 confirmation arrays. It does not recompute features under a wrong beta. Instead, it transfers the already-computed twelve-PC state contribution from a different whole matrix at the same candidate and landmark while keeping:

- the target state's original history inputs;
- the target state's existing outcomes;
- the frozen model and coefficients; and
- the empirical distribution of the transferred state block exactly unchanged.

For `M` matrices, every nonzero cyclic shift `m -> (m+k) mod M` is fixed-point-free. The analysis evaluated all `M-1` shifts:

- 39 shifts for the 40-matrix cohort; and
- 199 shifts for the 200-matrix cohort.

This is complementary to the coupling control:

- coupling derangement preserves the state's own matrix identity and beta invariants but shifts some conditional feature marginals;
- whole-matrix block derangement preserves the block marginals exactly but transfers matrix-specific content as well as local state content.

### 7.2 All cyclic shifts

| Cohort | Candidate | Half | Aligned gain | Minimum shifted gain | Median shifted gain | Maximum shifted gain | Positive shifts |
|---|---|---|---:|---:|---:|---:|---:|
| 40 matrices | 02 | A | +0.038851 | -0.148933 | -0.097431 | -0.054911 | 0/39 |
| 40 matrices | 02 | B | +0.035246 | -0.149943 | -0.095931 | -0.058868 | 0/39 |
| 40 matrices | 03 | A | +0.035028 | -0.249354 | -0.179075 | -0.123978 | 0/39 |
| 40 matrices | 03 | B | +0.034889 | -0.252715 | -0.168763 | -0.115712 | 0/39 |
| 200 matrices | 02 | A | +0.029188 | -0.162906 | -0.129607 | -0.096387 | 0/199 |
| 200 matrices | 02 | B | +0.028935 | -0.160168 | -0.127185 | -0.095144 | 0/199 |
| 200 matrices | 03 | A | +0.030869 | -0.210186 | -0.176113 | -0.129669 | 0/199 |
| 200 matrices | 03 | B | +0.033275 | -0.208058 | -0.174519 | -0.126720 | 0/199 |

Every exact-marginal pairing destroys the gain. Even the least damaging shifted pairing is worse than direct history.

### 7.3 Fixed shift-one descriptive intervals

For reference, the first fixed-point-free cyclic shift gives:

| Cohort | Candidate | Half | Shift-one gain | 95% whole-matrix CI |
|---|---|---|---:|---:|
| 40 matrices | 02 | A | -0.091990 | [-0.15338, -0.04083] |
| 40 matrices | 02 | B | -0.105900 | [-0.18786, -0.04257] |
| 40 matrices | 03 | A | -0.194700 | [-0.31750, -0.10167] |
| 40 matrices | 03 | B | -0.189604 | [-0.31424, -0.09266] |
| 200 matrices | 02 | A | -0.128404 | [-0.18102, -0.08327] |
| 200 matrices | 02 | B | -0.135609 | [-0.19396, -0.08933] |
| 200 matrices | 03 | A | -0.183891 | [-0.23597, -0.13773] |
| 200 matrices | 03 | B | -0.175860 | [-0.22978, -0.12927] |

These intervals are also post-hoc and use the temporary bootstrap stream described above.

## 8. Interpretation

### 8.1 What the diagnostics support

The results support the following statement:

> In independent test 1, the frozen composite's advantage over direct history required the 195-coordinate block to be correctly aligned with the natural state and catalytic matrix. Retaining the same 21-input model capacity while breaking that alignment eliminated and reversed the gain.

The two controls address complementary objections:

- the coupling rescore uses the same natural state and its own matrix, preserves beta invariants, and breaks molecular-label coupling;
- the exact-marginal rescore preserves the full empirical distribution of state-block values while breaking their assignment to outcomes.

Together, they make a generic additional-capacity explanation implausible for this implementation.

### 8.2 What the diagnostics do not support

They do not establish that:

- one specific network-state interaction has been isolated;
- beta-conditioned state is the unique source of the gain;
- composition, static beta, phase, and correlated history proxies have been separated;
- the extra state block is causal;
- the original L53/L54 comparison has been capacity-controlled;
- independent test 2 has passed the same rescore; or
- the analysis was preregistered.

“Content matters” must therefore mean **correctly aligned predictive content**, not “a unique physical mechanism has been identified.”

### 8.3 Why the result is positive for the preprint

The reviewer raised a potentially damaging alternative: the composite may win merely because it has more regularized inputs. The observed pattern is the opposite of that alternative's simplest prediction. The richer frozen model performs well only when its state block is correctly paired; wrong or reassigned content makes it worse than the smaller direct comparator.

The concern is nevertheless valuable because it forces the paper to distinguish three claims:

1. **Algorithmic advantage:** strongly supported across implementations.
2. **Advantage beyond generic fitted capacity:** supported by the present post-hoc controls in independent test 1.
3. **Physical attribution of the aligned information:** still unresolved.

## 9. Limitations and evidential status

1. **Reviewer-prompted and post hoc.** The rescoring question was designed after the original predictive results were known. The coupling permutation itself predates GN1 outcomes, but its use as a capacity control does not.
2. **One exact coupling implementation.** The primary recomputed-block analysis uses independent test 1's 195-coordinate representation and frozen 200-matrix model.
3. **Originating archive absent.** The source code and reports remain, but the exact L53/L54 model/state/outcome archive needed for an originating rescore is not in this checkout.
4. **Independent test 2 not rescored.** Its main-path features can be reconstructed without new futures, but that analysis has not been executed.
5. **No retained production script yet.** The numbers in this report were obtained from temporary read-only diagnostics and independently cross-checked against archived aligned point estimates. They should not enter the manuscript until a permanent, tested script and machine-readable artifact reproduce them.
6. **Temporary inference stream.** The new confidence intervals use 4,096 whole-matrix bootstrap draws from seed `20260818`, not a sealed preregistered stream.
7. **No multiplicity adjustment.** The descriptive intervals are candidate/half specific. A formal inferential family must be specified before manuscript promotion.
8. **Coupling-feature shift.** The coupling operation does not preserve every composition-conditioned feature marginal, although it does not increase extreme five-SD rows. The exact-marginal sensitivity is therefore essential context.
9. **Exact-marginal sensitivity removes more content.** Moving the full state block across matrices removes matrix propensity as well as local alignment. It is a capacity diagnostic, not a mechanistic analogue of acute within-matrix coupling derangement.

## 10. Recommended permanent analysis package before editing the preprint

Create an additive, read-only analysis package without modifying any sealed result bundle.

### 10.1 Suggested files

```text
replicators.13.8.2026.codex/
  REVIEWER_CAPACITY_MATCHING_PREREGISTRATION.md
  scripts/run_capacity_matching_rescore.py
  tests/test_capacity_matching_rescore.py
  results/capacity_matching_rescore/
    protocol.json
    result.json
    state_scores.csv.gz
    inference_arrays.npz
    readback_audit.json
    SHA256SUMS
    REPORT.md
```

Because the analysis question is already known, the protocol must call itself a **frozen post-hoc analysis plan**, not a prospective preregistration in the same sense as an untouched experiment.

### 10.2 Required integrity checks

The permanent script should:

1. verify every source checksum;
2. verify the frozen model SHA-256 shown above;
3. verify state-table and array state-ID order;
4. reconstruct all archived aligned predictions to tolerance `<=1e-12`;
5. verify that direct predictions are byte-identical before and after derangement;
6. verify that the derangement is fixed-point-free for every matrix;
7. verify exact preservation of beta's weight multiset, singular values, and other registered matrix invariants;
8. record feature- and PCA-space distribution-shift diagnostics;
9. use identical whole-matrix bootstrap indices for aligned, deranged, and their paired difference;
10. retain candidate and branch-half separation;
11. write all state-level predictions and losses;
12. reopen and validate every written artifact; and
13. run exact replay of the rescore itself.

### 10.3 Recommended estimands

For each candidate and half, report:

```text
aligned_gain  = LL(direct) - LL(aligned_full)
deranged_gain = LL(direct) - LL(deranged_full)
alignment_gain = LL(deranged_full) - LL(aligned_full)
```

The primary diagnostic should be the paired `alignment_gain`, because it directly asks whether correct block alignment improves the unchanged 21-input model. The `deranged_gain` answers the reviewer's simpler question of whether the richer model retains any advantage over direct history after alignment is broken.

If formal randomization values are added, define the four candidate-by-half cells and correction family before running the permanent script. Do not retrofit equivalence margins after seeing the present values.

## 11. Recommended preprint changes and exact locations

These are recommendations only. They have **not** been applied.

### 11.1 Methods: explicitly specify the direct comparator

**Location:** Chapter 2, under **Frozen F12 predictor and comparators**, immediately after the paragraph that ends with “ridge logistic regression with `C=0.1`.”

**Suggested text:**

> The direct-history comparator used the identical candidate-specific binomial L2-logistic pipeline—development-only imputation and standardisation, `lbfgs`, and `C=0.1`, with no hyperparameter search—but entered only the nine direct variables. The composite entered those same nine variables plus the twelve frozen PCA scores, giving nine versus 21 fitted slopes, with an intercept in each model. Thus the registered comparison matched estimator and regularisation setting but did not match representation dimension.

This sentence resolves the first part of the reviewer comment and states the remaining capacity issue directly.

### 11.2 Methods: add the reviewer-prompted capacity control

**Location:** Chapter 2, under **Generative-null design**, after the paragraph describing the 3,840 restored states and before the intervention-design section.

**Suggested heading:**

```markdown
### Reviewer-prompted predictive coupling rescore
```

**Suggested text:**

> After the original predictive and generative-null results were known, we specified a post-hoc, no-refit capacity diagnostic using the retained natural GN1 states and outcomes. For each natural launch state, we held its nine history variables and 64 observed F12 outcomes fixed, recomputed the 195-coordinate block after applying that matrix's already-sealed fixed-point-free simultaneous row/column permutation, and passed the result through the unchanged development scaler, PCA and 21-input logistic coefficients. The direct-history prediction was unchanged. Candidate and fixed branch halves remained separate, and uncertainty resampled whole catalytic matrices. No future, model, PCA object or calibration map was regenerated or refitted.
>
> Because acute coupling derangement can shift composition-conditioned feature marginals, a complementary exact-marginal sensitivity transferred the complete frozen state-block contribution among whole matrices at the same candidate and landmark. All nonzero cyclic matrix shifts were evaluated, preserving the empirical block distribution exactly while breaking its assignment to the original outcomes. Both analyses are reviewer-prompted and post hoc.

If the exact-marginal sensitivity is omitted from the permanent package, omit its methods paragraph as well; do not report it from this review document alone.

### 11.3 Results: insert the capacity result inside the generative-null section

**Location:** Chapter 3, under **Generative nulls separated geometry from catalytic context**, immediately after the paragraph reporting natural frozen-composite gains of `0.0243-0.0256` and before the paragraph beginning “The sharpest separation concerned control.”

**Suggested heading:**

```markdown
#### Predictive gain required the aligned state–matrix block
```

**Suggested text:**

> The reviewer-prompted no-refit rescore retained the same natural states, history variables, outcomes, 21-input dimension, frozen PCA and fitted coefficients while recomputing only the 195-coordinate block under the pre-existing coupling derangement. The aligned composite improved branch log loss over direct history by `+0.0243` to `+0.0256` nats. With the deranged block, the gain reversed to `-0.0114` to `-0.0208` nats across the four candidate/half cells, and every descriptive 95% whole-matrix interval lay below zero. The deranged representation did not increase the number of states outside the development five-standard-deviation envelope. In the exact-marginal sensitivity, all 39 nonzero matrix shifts at the 40-matrix scale and all 199 shifts at the 200-matrix scale also produced negative gains. Thus the richer frozen pipeline did not improve merely by retaining twelve additional fitted coordinates: its advantage required those coordinates to be correctly aligned with the natural state and matrix. This result distinguishes aligned content from generic model capacity, but it does not identify which physical feature family carries that content.

Add a compact four-row table with aligned and deranged gains and intervals. The full shift ledger belongs in Appendix D or the result bundle rather than the main text.

### 11.4 Appendix D: replace the open generic-capacity sentence

**Location:** Appendix D, in the paragraph that currently ends with “a regularisation-and-encoding explanation still open.”

**Suggested replacement text:**

> These feature-family nulls do not show that the original 195-coordinate block contributes nothing, because they changed both representation and fitting sequence. The separate reviewer-prompted capacity rescore kept the original frozen 21-input algorithm intact and altered only state–matrix alignment. Its gain over direct history vanished and reversed under coupling derangement, and an exact-marginal block reassignment gave the same qualitative result. Generic additional fitted capacity is therefore disfavoured in this implementation. The physical identity of the required aligned information remains unresolved: it may reflect composition, stable matrix propensity, beta-conditioned state, or a regularised combination of correlated history, phase and catalytic-context signals. The result does not isolate a unique network–state interaction.

Appendix D should contain the four-cell primary table, the feature-envelope counts, and the exact-marginal shift summary.

### 11.5 Discussion: narrow, rather than remove, the attribution caveat

**Location:** Chapter 5, under **The F12 predictor is reproducible and actionable**, in the paragraph beginning “The generative-null cohort provided a further untouched natural-data check.”

The current ending says that the block may carry a specific signal “or it may work as a regularised combination” of correlated variables. That wording conflates generic extra capacity with aligned but correlated information.

**Suggested replacement ending:**

> We therefore do not yet know which physical information in the 195-coordinate block drives prediction. The capacity-matched rescore shows that the block's correct alignment with the natural state and matrix is required, disfavoring feature count or PCA capacity alone. The aligned information may nevertheless be a regularised combination of composition, phase, history-correlated state, matrix propensity and catalytic context rather than one isolated network–state interaction.

### 11.6 Limitations: add implementation and post-hoc scope

**Location:** Chapter 5, under **Limitations**, immediately after the existing generative-null-scope item.

**Suggested item:**

> **Capacity-control scope.** The predictive coupling rescore was reviewer-prompted and post hoc. It used one clean-room implementation's retained natural cohort and pre-existing derangements; the originating predictor and the other independent implementation have not yet received the same recomputed-block test. It separates correct feature alignment from generic model dimension in that implementation but does not isolate the physical source of the aligned signal.

If the originating and independent-test-2 analyses are completed before submission, update this limitation rather than deleting the post-hoc designation.

### 11.7 Data and code availability

After the permanent analysis package exists, add links to:

- the frozen post-hoc protocol;
- the analysis script and tests;
- the state-level aligned/deranged score table;
- the machine-readable inference result;
- the input/model hash manifest; and
- the readback/replay audit.

Do not cite this report as if it were the final analysis artifact.

### 11.8 Abstract and title

No title change is needed. No abstract change is recommended unless the same capacity control is repeated across the originating workflow and both clean-room implementations. The main paper and Appendix D are sufficient for the present one-implementation result.

## 12. Minimal defensible revision versus ideal revision

### Minimal defensible revision

1. Make the direct comparator specification explicit.
2. Produce and retain the permanent independent-test-1 rescore package.
3. Add the post-hoc methods paragraph.
4. Add the four-cell result and a compact Appendix D table.
5. Replace the generic-capacity caveat with the narrower aligned-content caveat.
6. Add the one-implementation/post-hoc limitation.

### Ideal revision

In addition to the minimal revision:

1. recover the originating L53/L54 model, states and outcomes;
2. rerun the exact coupling rescore there;
3. reconstruct independent-test-2 landmark features without regenerating futures;
4. run the same frozen rescore in that implementation;
5. use implementation-specific pre-existing or deterministically frozen derangements;
6. report all three implementations in one Appendix D table; and
7. then state across implementations that the gain requires aligned content rather than generic extra capacity.

## 13. Recommended claim language

### Supported now, after permanent reproduction

> In independent test 1, the frozen composite's proper-score advantage required the 195-coordinate block to be correctly aligned with the natural state and catalytic matrix; retaining the same fitted dimension with deranged or reassigned block content eliminated the gain.

### Not supported yet

Avoid saying:

- “capacity matching is confirmed across all implementations”;
- “the beta-state interaction is the source of the gain”;
- “the analysis was preregistered”;
- “all feature marginals were preserved by coupling derangement”;
- “Appendix D now identifies the mechanism”; or
- “the originating `0.041-0.052` nat result has been capacity-controlled.”

## 14. Final assessment

The reviewer comment should be treated as constructive and substantively correct. The manuscript currently omits the direct model specification and does not contain the requested capacity-matched control. The retained data, however, produce the result the preprint would hope to see: aligned state/network content is necessary for the frozen composite's gain, while additional dimension alone is not sufficient.

The scientifically appropriate revision is therefore not to defend the existing text unchanged. It is to:

- disclose the identical direct-model contract;
- add the no-refit derangement rescore;
- report that the gain reverses;
- retain the physical-attribution caveat; and
- label the analysis post hoc and one-implementation until replicated more broadly.

No manuscript edit should be made from this report until the permanent analysis package reproduces the values and retains its full provenance.
