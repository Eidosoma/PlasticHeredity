# Sequence-history comparator review report

**Date:** 2026-08-18  
**Status:** Review, retained-data audit, and read-only exploratory diagnostic only. No preprint, model, simulation, registration, or scientific result artifact was changed.  
**Scope:** Whether the frozen F12 composite predictor should be compared with a first-order Markov, semi-Markov, or richer sequence-history baseline in addition to the registered nine-variable history-only ridge comparator.

## Executive finding

The reviewer has identified a legitimate remaining robustness question, but the retained definitions and data make it less threatening to the preprint than the numerical comparison in the comment suggests.

1. This concern is connected to, but distinct from, the companion [capacity-matching concern](COMPOSITE_HISTORY_CAPACITY_MATCHING_REPORT.md). The capacity control asks whether twelve extra fitted coordinates help merely because they add model capacity. The present concern asks whether the correctly aligned state/network coordinates proxy sequence-history information omitted by the nine-variable representation.
2. Appendix C does not provide a direct competitor to the headline predictor. It predicts the next inheritance symbol from symbols already observed **inside a future trajectory after its first break**. The headline model predicts the complete F12 event from information available at the launch state.
3. The numerical magnitudes in the comment are not commensurate. F12 gains are nats per stochastic future; Appendix C gains are bits per scored transition. They concern different responses, conditioning information, supports, and observational units. The current manuscript already states the unit distinction explicitly.
4. The registered nine-variable comparator is not an IID or sequence-naive baseline. It contains the exact launch-state quantities used by first-order and duration-aware models: current inheritance state and current regime duration. It additionally contains prefix and recent-five inheritance fractions, trailing inherited run, latest continuous parent-daughter similarity, generation, mass, and time since the latest break.
5. In fact, the clean-room H9 definition contains enough redundancy to represent state-specific linear duration effects: `trailing_inheritance_run` and `fissions_since_latest_break` are identical under the registered definition, while `current_regime_duration` supplies the duration of a non-inherited run. This does not make H9 equivalent to a nonlinear or generative sequence model, but it makes it substantially stronger than the Appendix C IID baseline.
6. A read-only exploratory check on independent test 1 supports that distinction. Candidate-specific F12 event predictors using only the launch-time Markov state, or the state plus the registered `1,2,3,4,5+` duration bin, were substantially worse than H9 at both the retained 40- and 200-matrix scales. The frozen composite remained better than H9.
7. That check is encouraging but does **not** close the review item. It is a target-specific launch-state diagnostic, not a full transition model and not a model using the complete raw pre-launch sequence. A richer sequence model could still exploit order, nonlinear duration effects, or higher-order patterns omitted by H9.
8. A leakage-free rescore is feasible without generating new stochastic futures in both clean-room implementations. Some deterministic replay is needed because compact result arrays do not retain every raw pre-launch history or development transition sequence.
9. The originating L53/L54 raw artifacts are not present in this checkout. The originating workflow can receive the same control only if its original artifact archive is recovered or its upstream state and branch records are reconstructed.
10. The current preprint's narrow claim—improvement over the **registered nine-variable history-only ridge comparator**—remains accurate. It should not be broadened to improvement over history generally unless the stronger sequence comparison passes.

The net assessment is **cautiously positive for the preprint**. This is a worthwhile robustness analysis and a real limitation if left undone, but the Appendix C result does not presently explain away the composite advantage.

## Reviewer comment, verbatim

> The direct-history comparator may be too weak a baseline for the headline predictor claim. The composite's advantage is measured against a 9-feature ridge logistic on history (line 169). But Appendix C shows post-break sequences carry first-order Markov and (in test 1) semi-Markov structure worth 0.03–0.06 bits/transition. A referee will ask: does the frozen composite beat a sequence-model history baseline, not just the 9-scalar summary? Given the composite's gain is itself 0.03–0.05 nats, this is not a rhetorical question. If the retained artefacts allow scoring a Markov/semi-Markov history baseline on the confirmation cohorts without new simulation, I'd add it; if not, acknowledge the gap explicitly in limitation 7 or the predictor discussion (line 646).

## 1. Relationship to the capacity-matching concern

The two review items belong in one predictor-robustness discussion, but one cannot substitute for the other.

| Concern | Alternative explanation being tested | Suitable control |
|---|---|---|
| Fitted capacity | The 21-input composite wins simply because it has twelve more regularized coordinates and a development-fitted PCA | Keep the frozen 21-input pipeline and break only state/network-block alignment |
| History baseline adequacy | The aligned block wins because it proxies sequence history omitted by H9 | Fit and freeze a stronger history-only sequence model, then score the same confirmation outcomes |

The retained derangement controls are positive for the first question: the frozen composite requires correctly aligned block content. They do not identify the aligned information. A current composition can encode consequences of recent history, so a correctly aligned state block could still proxy omitted sequence structure.

The present reviewer comment therefore survives the capacity result in a narrower form:

> Does the frozen composite retain incremental predictive value over a development-fitted model that uses the pre-launch inheritance sequence more completely than H9?

That is the right version of the question.

## 2. What the current preprint establishes

### 2.1 Comparator and target

The current Methods section defines the F12 endpoint as:

> Within the next twelve fissions, observe an inheritance break and, strictly after that break, certify three consecutive inherited fissions.

The event is scored once per stochastic future. The predictor estimates a state-indexed probability before any of those future fissions are observed.

Under **Frozen F12 predictor and comparators**, the manuscript describes:

- nine past-observable history/phase variables;
- a 195-coordinate current-state/catalytic-context block;
- development-only scaling and 12-component PCA of that block; and
- ridge logistic regression with `C=0.1` on the twelve components plus the nine direct variables.

It then enumerates all nine history variables. The companion capacity report audits the registrations and code and confirms that the direct-only comparator is also standardized L2 logistic ridge with `C=0.1`, candidate-specific development fitting, and no hyperparameter search.

Relevant manuscript sections:

- [`../PRE_PRINT_PAPER_DRAFT.md`, “Frozen F12 predictor and comparators”](../PRE_PRINT_PAPER_DRAFT.md#frozen-f12-predictor-and-comparators)
- [`../PRE_PRINT_PAPER_DRAFT.md`, “A state-dependent probability and a frozen predictor”](../PRE_PRINT_PAPER_DRAFT.md#a-state-dependent-probability-and-a-frozen-predictor)
- [`../PRE_PRINT_PAPER_DRAFT.md`, “The F12 predictor is reproducible and actionable”](../PRE_PRINT_PAPER_DRAFT.md#the-f12-predictor-is-reproducible-and-actionable)

### 2.2 Current claim language is appropriately narrow

The current manuscript repeatedly says that the composite outperformed its **registered nine-variable history-only ridge comparator**. It does not formally claim optimality over every possible history-only predictor.

That wording matters. The existing evidence supports an algorithm comparison:

```text
frozen composite recipe > registered H9 ridge recipe
```

It does not yet support the broader statement:

```text
current state/catalytic context adds information beyond every adequate model of history
```

The predictor remains a prospectively validated algorithm even if a later history model narrows its advantage. The unresolved baseline affects interpretation of the source of predictive information, not the existence, calibration, reproducibility, or intervention utility of the already-frozen algorithm.

### 2.3 Current limitation language does not yet name this gap

Current limitation 3 says that the composite's physical compression is unresolved and that no Phi-r variant supplied shared state-local foresight. It does not say that the registered H9 ridge has not been compared with a full sequence-history model.

Current limitation 7 concerns externally enacted feedback, so the reviewer's suggested numbering no longer matches the present draft. If the analysis is not run, the appropriate location is current limitation 3 and the F12 predictor discussion.

## 3. Why Appendix C is not the requested predictor comparison

### 3.1 The two analyses predict different outcomes

| Property | Headline F12 predictor | Appendix C sequence analysis |
|---|---|---|
| Predicted response | Whether one complete future contains `break ... inherit, inherit, inherit` | The next binary inheritance symbol |
| Prediction time | At the restored launch state | At each scored transition inside the realized future suffix |
| Conditioning data | Past-observable launch state and history | Previous realized future symbol; additionally its realized run duration for semi-Markov |
| Future support | All F12 futures, including no-break futures | Transitions in suffixes strictly after the first future break; unusable suffixes excluded |
| Main score | Branch log loss, nats per stochastic future | Cross-fitted log loss gain, bits per transition |
| Scientific question | State-local event probability over a twelve-fission horizon | Statistical dependence between successive post-break boundaries |

Appendix C therefore demonstrates that future inheritance symbols are not IID. It does not show that the **pre-launch sequence** contains unexploited information about state-to-state F12 risk.

### 3.2 Direct use of the Appendix C sequence would leak future information

For a confirmation state at launch, the following are unknown:

- whether a break will occur;
- when it will occur;
- the first symbol after the break;
- each later future inheritance symbol; and
- the run duration accumulated within that future.

Appendix C predicts a transition after some of those quantities have been observed. A fair F12 baseline must instead integrate over them. It may estimate transition laws from development data, but its confirmation prediction must be calculated before any confirmation future is opened.

Using a confirmation branch's realized post-break symbols to improve the predicted probability of that same branch's F12 event would condition on the target-generating future and would not be a baseline for the headline task.

### 3.3 The score magnitudes cannot be compared numerically

The manuscript explicitly states:

> F12 log-loss gains are nats per stochastic future; Appendix C's sequence-model gains are bits per transition.

For reference, multiplication by `ln(2)` converts the reported first-order gains to approximately `0.0235–0.0426` nats **per transition**, and the independent-test-1 duration increment to approximately `0.0069–0.0075` nats **per transition**. That conversion does not make the quantities comparable because:

- a future can contribute multiple scored Appendix C transitions;
- no-break, empty-suffix, and singleton-suffix futures do not contribute in the same way;
- the transition target differs from the composite event target; and
- Appendix C conditions on information unavailable at launch.

The numerical overlap cited by the reviewer is a reasonable motivation for checking a stronger baseline, but it is not evidence that the baseline will erase the F12 gain.

## 4. The registered H9 comparator already contains Markov state

The nine variables are:

1. normalized generation;
2. current mass;
3. prefix inheritance fraction;
4. recent-five inheritance fraction;
5. trailing inheritance run;
6. latest parent-to-daughter cosine similarity;
7. fissions since the latest break;
8. current inheritance state; and
9. current regime duration.

A homogeneous first-order Markov model requires the current inheritance state. A duration-aware semi-Markov model additionally requires the duration of the current run. H9 contains both.

It also supplies broader summaries that neither Appendix C model uses:

- long-run inheritance prevalence;
- recent local prevalence;
- the continuous value of the most recent `H`, rather than only its thresholded bit;
- phase/generation; and
- mass.

The clean-room code documents two exact relationships:

- `fissions_since_latest_break == trailing_inheritance_run`; and
- when the current state is inherited, `current_regime_duration == trailing_inheritance_run`.

When the current state is a break, `current_regime_duration` instead records the length of the consecutive non-inherited run. Consequently, the combination of current state, trailing inherited run, and current regime duration permits different linear duration behavior for inherited and non-inherited states.

This is not equivalent to:

- duration bins with unrestricted cell probabilities;
- nonlinear duration hazards;
- a generative transition process integrated across twelve steps;
- higher-order sequence dependence; or
- a model of the complete ordered prefix.

It does mean that Appendix C's positive Markov-over-IID result cannot be treated as evidence that the H9 comparator omitted the previous state or duration altogether.

Sources:

- [`plastic_heredity/features.py`](plastic_heredity/features.py)
- [`../replicators.13.8.2026.fable/replication/features.py`](../replicators.13.8.2026.fable/replication/features.py)
- [`plastic_heredity/memory_models.py`](plastic_heredity/memory_models.py)

## 5. Read-only exploratory launch-state diagnostic

### 5.1 Purpose

Before recommending a full replay, a narrow read-only diagnostic asked:

> If F12 prediction is restricted to the exact launch-state variables used by first-order or duration-aware sequence models, does that already rival H9 or the composite?

This is not the full analysis requested by the reviewer. It is a quick check of whether the specific state and duration dependence established in Appendix C is an obvious explanation for the headline advantage.

### 5.2 Retained inputs

The diagnostic used independent test 1's immutable arrays:

- [`results/full/analysis_arrays.npz`](results/full/analysis_arrays.npz): 40 development matrices, 400 development states, 32 development futures per state; 40 confirmation matrices, 400 confirmation states, and 64 confirmation futures per state;
- [`results/full/confirmation_states.csv`](results/full/confirmation_states.csv);
- [`results/scaled5/analysis_arrays.npz`](results/scaled5/analysis_arrays.npz): 200 development matrices, 2,000 development states, 32 development futures per state; 200 confirmation matrices, 2,000 confirmation states, and 64 confirmation futures per state; and
- [`results/scaled5/confirmation_states.csv`](results/scaled5/confirmation_states.csv).

No future, feature, fitted composite, or registered H9 prediction was regenerated or changed.

### 5.3 Diagnostic models

Models were candidate-separated and fitted only to the retained development branch targets.

**Markov-state event diagnostic**

```text
p(F12 event | current inheritance state)
```

**Semi-Markov-state event diagnostic**

```text
p(F12 event | current inheritance state, min(current duration, 5))
```

The duration categories were the Appendix C categories `1,2,3,4,5+`. Each cell used a Beta(1,1)-smoothed development event rate. An unseen confirmation cell would fall back to the candidate-specific Beta(1,1)-smoothed development prevalence.

These models estimate the F12 event directly rather than fitting transition probabilities. Thus they are target-specific tests of the launch-state variables, not generative Markov chains. They receive no current composition, catalytic matrix, graph feature, continuous raw sequence, or confirmation outcome during fitting.

### 5.4 Confirmation log loss

All values below are nats per stochastic future, averaged over the two fixed confirmation halves.

| Cohort | Candidate | Markov-state diagnostic | Semi-Markov-state diagnostic | Registered H9 ridge | Frozen composite |
|---|---:|---:|---:|---:|---:|
| 40 matrices | 02 | 0.623189 | 0.634426 | 0.567651 | 0.530603 |
| 40 matrices | 03 | 0.645520 | 0.647811 | 0.581743 | 0.546785 |
| 200 matrices | 02 | 0.643769 | 0.624993 | 0.573636 | 0.544575 |
| 200 matrices | 03 | 0.662031 | 0.642478 | 0.593569 | 0.561497 |

Lower log loss is better. At the larger scale, duration information improved on the current-state-only diagnostic, as one might expect, but both launch-state diagnostics remained well behind H9.

Across all eight candidate-by-half cells:

- the composite improved over H9 by `0.0289–0.0389` nats per future; and
- the composite improved over the semi-Markov-state diagnostic by `0.0793–0.1051` nats per future.

### 5.5 Interpretation

This result is reassuring for the preprint for one specific reason:

> The first-order state and run-duration variables highlighted by Appendix C do not, by themselves, provide a competitive F12 predictor in the retained independent-test-1 cohorts; H9 already uses those variables more effectively together with its additional summaries.

It does **not** establish that the composite beats every sequence-history model.

### 5.6 Why this is not manuscript-ready evidence

The diagnostic was performed after the reviewer comment and after all headline outcomes were known. It has the following limitations:

1. It is post hoc and exploratory.
2. It was run in only one clean-room implementation.
3. It predicts the event directly from Markov state rather than fitting and integrating a transition model.
4. It does not use the complete ordered pre-launch sequence.
5. It does not model continuous lagged `H` values.
6. It does not test second- or higher-order sequence dependence.
7. It uses one fixed smoothing and duration-binning rule borrowed from Appendix C.
8. It has no whole-matrix bootstrap, randomization test, multiplicity adjustment, or sealed gate.
9. The poorer 40-matrix semi-Markov result illustrates sparse-cell estimation rather than evidence against duration dependence.

The values should therefore guide design and risk assessment, not be inserted into the manuscript as a completed control.

## 6. Retained-data feasibility audit

### 6.1 Originating L53/L54 workflow

The checkout retains the originating runner code and Markdown result reports. The L54 runner documents that the original campaign wrote:

- frozen L53 transformations and fitted models;
- state features;
- prediction tables;
- restored confirmation states;
- branch seed manifests; and
- confirmation branch outcomes.

The runner refers specifically to artifacts such as:

```text
L53/state_feature_results.parquet
L53/transformed_feature_results.parquet
L53/prediction_results.parquet
L53/fitted_model_registry.parquet
L54/branch_seed_manifest.parquet
L54/branch_results.parquet
```

Those machine-readable L53/L54 files are not present in this checkout; only their reports are retained under the local step-report tree. The original scripts also expect upstream `/artifacts` and `/cache` state unavailable here.

Consequences:

- an immediate originating-workflow sequence rescore is not possible locally;
- recovering the original archive would probably make it possible without new scientific futures; and
- absent recovery, reproducing it requires reconstructing upstream trajectories, restored states, development branches, and frozen split directions.

Relevant sources:

- [`../original.1.8.2026.eidosoma-ai-scientist.code/arrival-of-self-replicators-eidosoma-groups-42/scripts/e01/run_s19_l53_regime_capacity_proxy.py`](../original.1.8.2026.eidosoma-ai-scientist.code/arrival-of-self-replicators-eidosoma-groups-42/scripts/e01/run_s19_l53_regime_capacity_proxy.py)
- [`../original.1.8.2026.eidosoma-ai-scientist.code/arrival-of-self-replicators-eidosoma-groups-42/scripts/e01/run_s19_l54_untouched_process_risk_confirmation.py`](../original.1.8.2026.eidosoma-ai-scientist.code/arrival-of-self-replicators-eidosoma-groups-42/scripts/e01/run_s19_l54_untouched_process_risk_confirmation.py)
- [`../original.1.8.2026.eidosoma-ai-scientist.stepReports/artifacts/research_steps/S19/loops/L54/S19_L54_FULL_RESULTS.md`](../original.1.8.2026.eidosoma-ai-scientist.stepReports/artifacts/research_steps/S19/loops/L54/S19_L54_FULL_RESULTS.md)

### 6.2 Independent test 1

The compact 40- and 200-matrix archives retain:

- development and confirmation H9 features;
- development and confirmation 195-coordinate blocks;
- development and confirmation branch-level F12 outcomes; and
- frozen direct and composite models.

They do not retain the raw ordered pre-launch inheritance and `H` sequences or the full development branch transition sequences.

However, the simulator contract is exactly reproducible:

- [`plastic_heredity/experiment.py`](plastic_heredity/experiment.py) deterministically rebuilds each development and confirmation main lineage from the registered master seed;
- [`plastic_heredity/simulator.py`](plastic_heredity/simulator.py) stores complete prefix `inheritance` and `boundary_h` tuples in every `Snapshot`; and
- development branch seeds are deterministic functions of cohort, candidate, matrix, landmark, and branch.

A valid sequence analysis can therefore:

1. rebuild only the main lineages needed to recover launch prefixes;
2. exactly replay development F12 branches to recover transition sequences;
3. verify that their F12 targets match the retained `development_targets` arrays byte-for-byte;
4. fit and freeze sequence models on those development records; and
5. score the already-retained confirmation targets without replaying or generating confirmation futures.

At the 200-matrix scale, this entails exact replay of 64,000 development F12 branches, not a new 128,000-future confirmation campaign.

### 6.3 Independent test 2

Independent test 2 retains:

- compact confirmation outcomes and predictions at 40, 200, and approximately 1,000 matrices in `results`, `results_5x`, and `results_25x`;
- frozen model bundles;
- deterministic development and confirmation seed tags; and
- a separate 200-matrix `v2` cohort whose [`results_sensitivity/v2_cohort.pkl`](../replicators.13.8.2026.fable/replication/results_sensitivity/v2_cohort.pkl) contains all 64 future `H` sequences per state, their observed lengths, and the H9 and 195-coordinate features.

The compact tables generally do not retain complete pre-launch raw histories. They are reproducible from [`replication/cohort.py`](../replicators.13.8.2026.fable/replication/cohort.py):

- `matrix_and_init` and the main-path seed reproduce the launch trajectory;
- `dev_unit` reproduces development labels and features from the same main path; and
- `conf_features_unit` reproduces confirmation launch features without running branch futures.

For the strongest retained second-implementation test, the analysis could:

1. replay the 25x development main paths used to fit the v2 predictor;
2. extract each ordered prefix and the future sequence producing its development F12 label;
3. fit the sequence model on those development records only; and
4. score the existing 200-matrix v2 confirmation outcomes.

The retained `H64` confirmation sequences should be used only to verify target reconstruction or for separately declared diagnostics. They must not be used to fit a predictor for their own F12 outcomes.

### 6.4 Appendix C artifacts

Independent test 1 retains complete prospective memory artifacts:

- [`results/memory_confirmation/sequences.csv.gz`](results/memory_confirmation/sequences.csv.gz)
- [`results/memory_confirmation/model_fits.csv`](results/memory_confirmation/model_fits.csv)
- [`results/memory_confirmation/crossfit_losses.npz`](results/memory_confirmation/crossfit_losses.npz)
- [`results/memory_confirmation/MEMORY_RESULTS.md`](results/memory_confirmation/MEMORY_RESULTS.md)

They provide a useful frozen specification for:

- Beta(1,1) smoothing;
- `1,2,3,4,5+` duration bins;
- candidate separation;
- whole-matrix cross-fitting; and
- support-matched transition scoring.

They cannot simply be reused as the requested F12 baseline because their sequences begin strictly after the first future break and omit the transition from launch through first break. They therefore model recovery-side dependence but not the complete break-and-renewal probability.

## 7. Recommended primary analysis

### 7.1 Scientific estimand

The primary question should be:

> On the same confirmation states and outcomes, does the frozen composite improve branch log loss over a candidate-specific, development-fitted sequence-history model of the complete F12 process?

This is different from asking whether post-break transitions are Markovian.

### 7.2 Construct complete development sequences

For each development launch state and branch, construct:

```text
(I_0, I_1, ..., I_12)
```

where:

- `I_0` is the final inheritance state observed in the pre-launch main-lineage prefix; and
- `I_1 ... I_12` are the branch's future inheritance states.

For semi-Markov fitting, initialize the duration at the complete pre-launch run duration and update it after each simulated transition. Cap it only through the frozen `1,2,3,4,5+` binning rule.

All transitions should be retained, including those in no-break futures. Restricting fitting to post-break suffixes would fail to model the break component of F12.

### 7.3 Freeze nested sequence models

At minimum, fit:

1. **First-order Markov:**

   ```text
   P(I_{t+1}=1 | I_t)
   ```

2. **Duration-aware semi-Markov:**

   ```text
   P(I_{t+1}=1 | I_t, duration_bin_t)
   ```

Use a prespecified smoothing rule, preferably the existing Appendix C Beta(1,1) rule, and preserve candidate separation.

For the originating workflow, preserve the original A-to-B and B-to-A development/confirmation-half directions. For each clean room, preserve its own original development and confirmation contract rather than imposing a different training scheme retrospectively.

### 7.4 Convert transition laws into an F12 launch probability

The transition models must produce one probability per launch state before confirmation outcomes are used. This can be calculated exactly by dynamic programming.

A sufficient recursive state is:

```text
(last inheritance bit,
 current duration bin,
 future break already seen?,
 post-break trailing inherited-run length capped at 3)
```

At launch, `future break already seen` is false even when the historical prefix contains earlier breaks, because the registered endpoint requires a break inside the next twelve fissions.

Propagate the fitted transition probabilities for twelve steps. The success state is absorbing once a future break has been followed strictly later by three consecutive inherited transitions.

This avoids Monte Carlo error and prevents any confirmation future symbol from entering its own prediction.

### 7.5 Score the same confirmation outcomes

For every candidate and fixed branch half, report:

- H9 log loss;
- Markov F12 log loss;
- semi-Markov F12 log loss;
- composite log loss;
- semi-Markov gain over H9; and
- composite gain over the strongest history-only baseline.

Use paired whole-matrix resampling and retain the manuscript's nats-per-future convention. Candidate and branch-half results should remain separate.

The most informative primary contrast is:

```text
LL(strongest frozen history-only sequence model)
  - LL(frozen composite)
```

A positive value favors the composite.

### 7.6 Integrity requirements

Before scoring confirmation outcomes, the implementation should verify:

1. candidate and matrix identities match retained arrays;
2. regenerated H9 features match archived H9 features exactly or within an advance floating-point tolerance;
3. regenerated development F12 labels match retained labels exactly;
4. no confirmation transition or outcome enters model fitting;
5. no hyperparameter, duration bin, sequence order, or smoothing rule is selected from confirmation performance;
6. registered H9 and composite predictions remain byte-identical;
7. predictions exist for every retained confirmation state;
8. all inference resamples whole catalytic matrices; and
9. the analysis is labelled reviewer-prompted and post hoc.

## 8. Recommended stronger sensitivity

The first-order/semi-Markov analysis responds directly to the reviewer, but it does not use the complete ordering of the pre-launch prefix beyond its current state and duration. A fuller sensitivity would test whether H9's scalar summaries discard useful sequence structure.

One defensible design is a development-selected but confirmation-frozen history-only model containing:

- a fixed recent window of binary inheritance lags;
- the same window of continuous `H` lags;
- explicit masks for unavailable early lags;
- current run duration and state;
- the non-sequence phase and mass variables already in H9; and
- a fixed regularized estimator.

Sequence-window length, interactions, and regularization must be fixed from development-only cross-validation or an advance rule. Trying many recurrent, tree, convolutional, and lag-window models against the already-viewed confirmation results and reporting only the best would replace one baseline concern with model-selection optimism.

This sensitivity answers a broader question than a homogeneous semi-Markov model:

> Does the composite add predictive value beyond a reasonably expressive representation of the complete observed prefix?

It should be secondary unless its design is sealed before results are opened.

## 9. Possible outcomes and interpretation

### 9.1 Composite clearly beats the sequence baseline

This would materially strengthen the preprint:

- the registered H9 result remains;
- the capacity derangement shows that aligned block content is required; and
- the sequence comparison shows that the gain is not reproduced by first-order or duration-aware history dynamics.

The claim could then be:

> In the tested implementations, the frozen composite improved F12 prediction over both the registered scalar-history comparator and a development-fitted Markov/semi-Markov history model.

It would still not identify a unique physical mediator.

### 9.2 Sequence baseline improves over H9 but composite remains better

This is also a positive result. It would show that H9 was not maximally strong while preserving incremental composite value. The manuscript should report both facts rather than describing H9 as sufficient.

### 9.3 Sequence baseline matches the composite

The algorithmic result remains valid—composite beat the comparator against which it was prospectively registered—but the physical interpretation narrows. The aligned state/network block may be acting largely as a proxy for omitted history dynamics.

The preprint should then avoid implying that current composition or catalytic context adds information beyond history generally.

### 9.4 Sequence baseline beats the composite

This would require a substantive narrative revision but would not erase:

- state-dependent F12 probability;
- exact replication of the frozen composite algorithm;
- intervention effects selected by that algorithm;
- repeated-feedback control; or
- strict coherent-event occurrence.

It would show that the headline predictor comparison used a weak history representation and that an explicit sequence model is the better predictive coordinate.

## 10. Manuscript recommendations if the analysis is run

No changes in this section have been applied.

### 10.1 Methods

Insert a short subsection immediately after **Frozen F12 predictor and comparators**.

Suggested structure:

> **Reviewer-prompted sequence-history comparator.** After the registered predictor results were known, we specified a post-hoc, no-new-future comparison with development-fitted first-order and duration-aware history models. For every development branch, the fitting sequence joined the final pre-launch inheritance state to all twelve future boundary states. The first-order model estimated the next inheritance probability conditional on the previous state; the duration-aware model additionally used a frozen `1,2,3,4,5+` past-only run-duration bin, with Beta(1,1) smoothing. Candidate-specific parameters were fitted only on the original development records. Exact dynamic programming then converted each frozen transition law and launch-state history into the probability of a future break followed strictly later by three inherited boundaries within F12. No confirmation transition, outcome, PCA object, composite coefficient, or H9 prediction was used or refitted.

The final text must reflect the actual implementation and local split contracts.

### 10.2 Results

Place the main result after the cross-clean-room F12 predictor comparison. A compact table should give, by implementation and candidate/half:

- H9 loss;
- Markov loss;
- semi-Markov loss;
- composite loss; and
- composite-minus-strongest-history gain with a whole-matrix interval.

Keep complete cell and replay details in Appendix D or a dedicated supplementary result bundle.

### 10.3 Appendix C

Add an explicit bridge distinguishing the tasks:

> Appendix C's gains concern next-transition prediction conditional on realized symbols after a future break and are reported in bits per transition. They are not themselves launch-time predictors of the compound F12 event and cannot be compared numerically with nats per stochastic future. The separate sequence-history comparator fits complete transition laws on development records and integrates over unobserved future symbols before scoring confirmation outcomes.

### 10.4 Predictor discussion

If the composite passes:

> The composite's advantage was not limited to the registered nine-variable ridge. It also improved proper score over development-fitted first-order and duration-aware sequence-history baselines that integrated the complete F12 event from the launch state. This constrains an omitted Markov/run-duration explanation, while leaving higher-order history and physical feature attribution open.

If only one implementation is tested, state that scope explicitly and do not generalize across all three workflows.

### 10.5 Limitations

If the full sequence comparison is not run, add to current limitation 3:

> **History-baseline scope.** The registered comparator summarizes launch history in nine variables and includes current inheritance state and run duration, but it is not a model of the complete ordered prefix. Appendix C establishes post-break next-transition dependence on realized future symbols and therefore does not supply a leakage-free launch-time F12 comparator. A development-fitted full sequence-history baseline remains untested.

If only the Markov/semi-Markov comparison is run:

> The reviewer-prompted sequence comparison was post hoc and tested first-order and duration-aware dynamics, not every higher-order or nonlinear history model. The originating workflow lacked the local raw artifacts required for the same rescore.

### 10.6 Abstract and title

No title change is indicated.

Do not add a sequence-baseline claim to the abstract unless it is reproduced across both clean rooms under one clearly stated comparison. A one-implementation or exploratory result belongs in the main predictor discussion and limitations.

## 11. Manuscript recommendation if no analysis is run

If artifact recovery or replay is deferred, the preprint should make two narrow clarifications:

1. Appendix C does not furnish a sequence-model comparator for F12 because it conditions on post-break future transitions and uses a different score unit.
2. The composite's validated advantage is over the registered H9 ridge, not over all possible history-only models.

Suggested discussion sentence:

> The registered history comparator includes current inheritance state, run duration, recent and prefix inheritance fractions, and the latest continuous parent-daughter similarity, but it compresses the ordered prefix into nine scalars. We have not yet established incremental value over a development-fitted full sequence-history model.

This acknowledgment would be scientifically adequate, although less persuasive than the retained-data rescore.

## 12. Priority and risk assessment

| Dimension | Assessment |
|---|---|
| Scientific validity of reviewer concern | Real but narrower than stated |
| Validity of direct numerical comparison | Low; units, targets, and conditioning differ |
| Evidence that H9 entirely omits Markov/duration information | None; those variables are explicit in H9 |
| Preliminary retained-data direction | Positive for the preprint |
| Remaining possibility of a stronger raw-sequence explanation | Open |
| Feasibility without new confirmation futures | High in both clean rooms |
| Feasibility in originating workflow from this checkout | Blocked pending artifact recovery |
| Likelihood of erasing all composite gain | Appears low to moderate, not zero |
| Reviewer vulnerability if omitted and unacknowledged | Moderate |
| Effect on intervention and feedback results | Limited; those results do not depend on H9 being the strongest possible baseline |

Recommended priority among predictor analyses:

1. preserve and formally package the capacity-matching rescore;
2. specify the leakage-free Markov/semi-Markov F12 baseline;
3. run it on the two clean-room retained cohorts;
4. add a complete-prefix sensitivity if a defensible development-only model-selection rule can be frozen; and
5. recover the originating L53/L54 archive before claiming three-workflow coverage.

## 13. Bottom line

The reviewer is correct that a stronger sequence-history comparator would improve the paper. The comment should not, however, be read as showing that the present H9 baseline is equivalent to IID history or that Appendix C's `0.03–0.06` bits per transition can be set directly against the composite's `0.03–0.05` nats per future.

The strongest evidence presently available is:

- H9 already contains the first-order state and duration variables highlighted in Appendix C;
- Appendix C uses future-conditioned transition prediction rather than launch-time event prediction;
- a read-only launch-state diagnostic is substantially worse than H9 in both retained independent-test-1 cohorts; and
- the composite remains approximately `0.029–0.039` nats per future better than H9 in those diagnostic cells.

The unresolved question is whether a **complete, ordered, nonlinear pre-launch history model** can close that remaining gap. That question is answerable in both clean rooms without new stochastic futures, but it has not yet been answered.

Accordingly, the current result remains positive and accurately stated as an advantage over the registered H9 ridge. A clean sequence-history rescore would strengthen the headline; absent that rescore, the scope should be acknowledged explicitly.

