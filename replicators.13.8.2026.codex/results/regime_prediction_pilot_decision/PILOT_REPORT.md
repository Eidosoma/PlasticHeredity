# Strict-regime prediction pilot: expanded decision report

**Report status:** Post-hoc decision support, written 15 August 2026 from the checksum-sealed pilot artifacts.

**Sealed pilot status:** `stopped_before_confirmation`.

**Registered decision:** **Failed. No confirmation is authorized.**

**Why this is a companion report:** The canonical pilot bundle was sealed on 14 August 2026 at 20:06 UTC. Its original `PILOT_REPORT.md` and `SHA256SUMS` remain byte-for-byte unchanged. This expanded report is stored separately so that interpretation added after seeing the outcome cannot be mistaken for a preregistered artifact.

## Executive summary

The pilot executed correctly. It generated 800 fresh restored states from 80 catalytic matrices, shot 102,400 independent F32 futures, regenerated all 102,400 futures, reproduced every discrete and continuous output exactly, and recorded enough strict events to pass both prospective event-power gates. The scientific stop was therefore not caused by a software error, failed replay, too few events, or incomplete trajectories.

Four of the six model families improved strict-event log loss over the complete unique-history/all-clock `h10` baseline in both simulator candidates and both fixed branch halves. This is encouraging development evidence that the present state/network observables contain predictive information. However, the registered question was stricter: one common family had to be selected stably in at least 75% of 4,096 paired catalytic-matrix bootstraps. The bootstrap winner split mainly between hurdle modeling (`55.44%`) and direct ridge (`43.63%`). Neither reached 75%, so no family was frozen.

The split is scientifically informative. Direct ridge was stronger in candidate 02, whereas the hurdle decomposition was stronger in candidate 03. The guarded nonlinear model made this heterogeneity more extreme: it hurt both candidate-02 halves but produced the largest gains in candidate 03. Thus the pilot suggests real predictability but does not identify one representation or mechanism that transfers cleanly across both simulator implementations.

The existing confirmation command must not be run. If prediction is the priority, the cleanest economical next experiment is a new, separately registered hypothesis that freezes an equal-probability ensemble of direct ridge and hurdle before generating a fresh 200-matrix cohort. That ensemble was devised after seeing this pilot and therefore cannot rescue the failed registration; only a new untouched cohort could test it. If a common mechanism is the priority, a larger fresh development cohort followed by another untouched confirmation is more rigorous but substantially more expensive.

## 1. Question and registered endpoint

The pilot asked whether observables available at a restored post-fission state predict the probability of a strict break-and-distinct-renewal event better than direct history and all growth clocks alone.

The unconditional primary event required all of the following within the next 32 fissions:

1. The first inheritance break, defined by parent-daughter cosine similarity `H <= 0.90`.
2. A later window of eight consecutive inherited fissions, each with `H > 0.90`.
3. Every pair among the eight daughters having cosine similarity `> 0.90`.
4. Every daughter having cosine similarity `<= 0.85` to the parent composition at the first break.

Every eligible post-break eight-run was searched. The binary endpoint was exactly equivalent to a positive best continuous joint margin. First-five coherence, centroid coherence, hurdle stages, continuous margins, and post-break models were registered as secondary analyses that could not rescue a failed primary selection.

## 2. Frozen design

| Item | Frozen value |
|---|---:|
| Catalytic matrices | 80 |
| Simulator candidates | 2 (`02`, `03`) |
| Restored landmarks per matrix/candidate | 5 (`20`, `35`, `50`, `65`, `80`) |
| Restored states | 800 |
| Futures per state | 128 |
| Primary futures | 102,400 |
| Full replay futures | 102,400 |
| Future horizon | 32 fissions |
| Whole-matrix cross-validation folds | 5 |
| Selection bootstraps | 4,096 |
| Bootstrap unit | One paired catalytic-matrix draw shared across candidates and families |
| Minimum events per candidate | 100 |
| Minimum event-positive matrices per candidate | 20 |
| Required family-selection frequency | 75% |

The baseline was the unpenalized, duplicate-cleaned `h10` history/all-clock block. Added raw feature blocks were state-only composition, complete beta-only structure without PCA, state-beta interactions, and analytic local dynamics. The fixed model menu was:

1. Direct sequential ridge.
2. Three-stage hurdle: break, later run8 conditional on break, and strict geometry conditional on run8.
3. Hierarchical beta-propensity offset followed by history, state, and dynamics.
4. Local-dynamics offset model.
5. Leakage-safe first-five/centroid auxiliary stack.
6. Bounded histogram-gradient model with out-of-fold beta propensity and Platt calibration.

The registered selection sequence was:

1. Stop if either event-power gate failed.
2. Exclude a family unless its enhanced-over-`h10` log-loss gain was positive in candidate 02 half A, candidate 02 half B, candidate 03 half A, and candidate 03 half B.
3. Find the lowest candidate-equal cross-fitted loss among eligible families.
4. Retain eligible families within one standard error of that loss.
5. Choose the first retained family in the registered simplicity order.
6. Require that provisional choice to win at least 75% of paired whole-matrix selection bootstraps.

## 3. Execution and integrity audit

| Check | Result |
|---|---:|
| Service exit | Success (`ExecMainStatus=0`) |
| Campaign phase | `sealed_complete` |
| State generation checkpoints | 800/800 |
| Replay checkpoints | 800/800 |
| Primary futures | 102,400/102,400 |
| Replay futures | 102,400/102,400 |
| Completed 32-fission horizons | 100% in both candidates |
| Discrete replay | Exact |
| Continuous values compared | 7,794,023 |
| Maximum continuous replay error | `0.0` |
| Replay digests | Identical |
| Replay digest | `059b870ec9ad68ecacabc59fa6bb9b972184ecfa9086e5359da71f5955ab267f` |
| Pilot checksum verification | Passed |
| Registration verification | Passed |

The registration ID was `19d9cd4c59a884928c33cd4028f57f8f6d012e6e75a23168f5bd0a795b800a2e`. The pilot seal ID is `4db89c5095682c4cf055ed0cb26f9ba80972fd849e8f9a722fe6edf84b3b08a7`. The sealed `SHA256SUMS` digest is `664587f1b8e73f9154790d93846ad64903f00a65b979c610612fee5d46264ad2`.

## 4. Event incidence and power

### 4.1 Overall stage and endpoint counts

| Candidate | Futures | Break | Later run8 | First-five | Centroid | Strict |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 51,200 | 36,655 (71.59%) | 27,586 (53.88%) | 3,841 (7.50%) | 3,840 (7.50%) | 1,190 (2.324%) |
| 03 | 51,200 | 38,748 (75.68%) | 28,846 (56.34%) | 4,298 (8.39%) | 4,322 (8.44%) | 1,292 (2.523%) |

The strict event remained rare, but it was observed more than ten times as often as the minimum event requirement in each candidate.

### 4.2 Conditional bottlenecks

| Candidate | Run8 given break | Strict given break | Strict given run8 | First-five given run8 | Centroid given run8 |
|---|---:|---:|---:|---:|---:|
| 02 | 75.26% | 3.246% | 4.314% | 13.92% | 13.92% |
| 03 | 74.45% | 3.334% | 4.479% | 14.90% | 14.98% |

The bottleneck was again geometric. Breaks and later eight-runs were common, but only about 4.3–4.5% of later eight-runs were both mutually coherent and old-anchor-distinct.

### 4.3 Fixed branch halves and matrix support

| Candidate | Half | Strict events | Futures | Rate | Matrices with an event |
|---|---|---:|---:|---:|---:|
| 02 | A | 603 | 25,600 | 2.355% | 45/80 |
| 02 | B | 587 | 25,600 | 2.293% | 47/80 |
| 03 | A | 641 | 25,600 | 2.504% | 47/80 |
| 03 | B | 651 | 25,600 | 2.543% | 52/80 |

Across both halves, candidate 02 had event-positive futures in 52 matrices and candidate 03 in 58 matrices. Both exceeded the registered 20-matrix minimum. Similar half rates also show that the failure was not caused by one anomalous shooting half.

### 4.4 Rates by restored landmark

| Candidate | Landmark | Break | Later run8 | Strict | First-five | Centroid |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 20 | 71.62% | 53.97% | 2.305% | 7.578% | 7.461% |
| 02 | 35 | 72.26% | 55.19% | 2.256% | 7.588% | 7.617% |
| 02 | 50 | 71.25% | 53.35% | 2.178% | 7.266% | 7.236% |
| 02 | 65 | 71.66% | 53.86% | 2.207% | 7.031% | 7.354% |
| 02 | 80 | 71.17% | 53.03% | 2.676% | 8.047% | 7.832% |
| 03 | 20 | 75.79% | 56.45% | 2.529% | 8.369% | 8.555% |
| 03 | 35 | 75.86% | 57.15% | 2.451% | 8.750% | 8.730% |
| 03 | 50 | 75.21% | 56.09% | 2.275% | 7.793% | 8.125% |
| 03 | 65 | 76.16% | 56.87% | 2.744% | 8.418% | 8.770% |
| 03 | 80 | 75.38% | 55.15% | 2.617% | 8.643% | 8.027% |

There was no single landmark at which the event disappeared or dominated. This supports retaining multiple landmarks in any follow-up.

### 4.5 Trajectory descriptives

| Quantity | Candidate 02 | Candidate 03 |
|---|---:|---:|
| Median first-break index, given a break | 6 | 5 |
| Median first later-run8 start, given a run8 | 11 | 11 |
| Median longest post-break inherited run | 11 | 11 |
| 90th percentile longest post-break run | 23 | 22 |
| Mean eligible eight-windows, given any | 8.57 | 8.39 |
| 90th percentile eligible windows | 17 | 17 |
| Maximum eligible windows | 24 | 24 |
| Median best strict margin, given a run8 | -0.3644 | -0.3484 |
| Median best strict margin, given a strict event | 0.0377 | 0.0356 |

Most later eight-runs missed the joint geometry threshold by a substantial margin. Positive events were not merely numerical boundary cases: their median positive margins were approximately 0.036–0.038.

## 5. Baseline performance

The baseline prediction was generated out of fold using complete unique direct history plus prior-cycle and cumulative growth clocks. Log losses are natural-log cross-entropies; lower is better.

| Candidate | Half | Event rate | Mean baseline prediction | Baseline log loss | Baseline Brier | Spearman | Matrix-centered Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| 02 | A | 2.355% | 2.504% | 0.108448 | 0.003585 | 0.272 | 0.086 |
| 02 | B | 2.293% | 2.504% | 0.106359 | 0.003371 | 0.285 | 0.067 |
| 03 | A | 2.504% | 2.723% | 0.128470 | 0.005310 | 0.087 | -0.023 |
| 03 | B | 2.543% | 2.723% | 0.129113 | 0.005345 | 0.147 | 0.150 |

Mean baseline calibration was reasonable, but ranking—especially within catalytic matrices—was weak. This left room for added state/network information without implying that the baseline was badly misspecified in prevalence.

## 6. Registered model-selection results

### 6.1 Primary selection table

| Family | Candidate-equal loss | 02 A gain | 02 B gain | 03 A gain | 03 B gain | Positive in all four? | Bootstrap wins |
|---|---:|---:|---:|---:|---:|---|---:|
| Direct ridge | 0.112281 | 0.006568 | 0.006000 | 0.004905 | 0.005793 | Yes | 1,787/4,096 (43.63%) |
| Hurdle | **0.111579** | 0.004345 | 0.002857 | 0.009044 | 0.009827 | Yes | **2,271/4,096 (55.44%)** |
| Hierarchical offset | 0.113533 | 0.003749 | 0.003385 | 0.005396 | 0.005727 | Yes | 36/4,096 (0.879%) |
| Local dynamics | 0.114910 | 0.003034 | 0.002644 | 0.003308 | 0.003762 | Yes | 2/4,096 (0.049%) |
| Auxiliary stack | 0.115742 | 0.000094 | -0.000102 | 0.004945 | 0.004485 | No | Ineligible |
| Guarded nonlinear | 0.113969 | -0.003943 | -0.004432 | 0.012970 | 0.011921 | No | Ineligible |

The gains are baseline log loss minus enhanced log loss, so positive values favor the enhanced family. Auxiliary and nonlinear frequencies are zero by construction because they failed the four-cell positivity requirement and were excluded before bootstrap selection.

The hurdle family had the lowest candidate-equal loss. Its standard error was `0.018966`, making the registered one-standard-error threshold `0.130545`. Direct ridge, hurdle, hierarchical offset, and local dynamics were all within that threshold. The simplicity order therefore made direct ridge the provisional choice. Direct ridge then won only 43.63% of paired bootstraps, below the required 75%. Hurdle was the most frequent bootstrap winner but also remained far below 75%.

### 6.2 Relative improvement over baseline

| Family | 02 A | 02 B | 03 A | 03 B |
|---|---:|---:|---:|---:|
| Direct ridge | 6.06% | 5.64% | 3.82% | 4.49% |
| Hurdle | 4.01% | 2.69% | 7.04% | 7.61% |
| Hierarchical offset | 3.46% | 3.18% | 4.20% | 4.44% |
| Local dynamics | 2.80% | 2.49% | 2.57% | 2.91% |
| Auxiliary stack | 0.09% | -0.10% | 3.85% | 3.47% |
| Guarded nonlinear | -3.64% | -4.17% | 10.10% | 9.23% |

This table exposes the main source of instability. Candidate 02 favored direct ridge. Candidate 03 favored hurdle, with nonlinear fitting producing an even larger but candidate-specific improvement.

### 6.3 Pooled candidate losses and calibration descriptives

| Family | Candidate-02 loss | Candidate-03 loss | Mean prediction 02 | Mean prediction 03 |
|---|---:|---:|---:|---:|
| Direct ridge | **0.101120** | 0.123442 | 2.356% | 2.581% |
| Hurdle | 0.103803 | 0.119355 | 2.433% | 2.575% |
| Hierarchical offset | 0.103836 | 0.123229 | 2.299% | 2.577% |
| Local dynamics | 0.104565 | 0.125256 | 2.414% | 2.672% |
| Auxiliary stack | 0.107407 | 0.124076 | 2.413% | 2.665% |
| Guarded nonlinear | 0.111591 | **0.116346** | 2.402% | 2.421% |

The observed strict rates were 2.324% and 2.523%. Mean predictions were generally close to prevalence. The problem was not gross calibration; it was transferable ranking and model-family stability.

### 6.4 Matrix-level breadth

| Family | 02 A positive matrices | 02 B | 03 A | 03 B |
|---|---:|---:|---:|---:|
| Direct ridge | 55/80 | 52/80 | 51/80 | 53/80 |
| Hurdle | 51/80 | 43/80 | 52/80 | 50/80 |
| Hierarchical offset | 51/80 | 49/80 | 50/80 | 49/80 |
| Local dynamics | 52/80 | 46/80 | 47/80 | 48/80 |
| Auxiliary stack | 31/80 | 32/80 | 47/80 | 42/80 |
| Guarded nonlinear | 35/80 | 34/80 | 49/80 | 43/80 |

Every family helped some matrices and hurt others. Empirical matrix-gain 2.5th-to-97.5th percentile ranges crossed zero in every cell. Thus even the positive aggregate gains should not be described as a universal matrix-level effect.

## 7. Fitted feature and regularization audit

### 7.1 Feature retention

| Block | Raw coordinates | Candidate 02 retained | Candidate 03 retained |
|---|---:|---:|---:|
| `h10` | 10 | 9 | 10 |
| State-only | 26 | 11 | 12 |
| Complete beta-only | 309 | 239 | 239 |
| State-beta interaction | 64 | 64 | 64 |
| Local dynamics | 106 | 94 | 95 |

Only development-fold constant or affine-duplicate coordinates were removed. No PCA was used. Candidate 02 lost one `h10` direction because it was constant or redundant in that cohort.

### 7.2 Main ridge penalties

| Family/block | Candidate 02 lambda | Candidate 03 lambda |
|---|---:|---:|
| Direct state | 100 | 0.01 |
| Direct beta | 1 | 1 |
| Direct interaction | 0.01 | 0.1 |
| Hierarchical state | 0.1 | 0.01 |
| Hierarchical beta | 1 | 1 |
| Hierarchical dynamics | 10 | 1 |
| Local beta | 1 | 1 |
| Local dynamics | 10 | 1 |

All `h10` baseline stages were unpenalized (`lambda=0`). The sharp candidate difference in selected state and dynamics penalties is consistent with the observed family instability; it does not by itself identify a mechanism.

### 7.3 Hurdle penalties

| Candidate | Stage | State | Beta | Interaction | Dynamics |
|---|---|---:|---:|---:|---:|
| 02 | Break | 0.1 | 1 | 0.01 | 100 |
| 02 | Run8 given break | 1 | 0.1 | 0.1 | 100 |
| 02 | Strict given run8 | 0.001 | 1 | 0.1 | 10 |
| 03 | Break | 1 | 1 | 0.1 | 1 |
| 03 | Run8 given break | 1 | 1 | 0.1 | 1 |
| 03 | Strict given run8 | 0.01 | 1 | 1 | 1 |

Candidate 02 heavily shrank dynamics in the break and run8 stages, whereas candidate 03 retained dynamics much more strongly throughout.

### 7.4 Auxiliary and nonlinear choices

For candidate 02, the relaxed first-five and centroid dynamics blocks selected `lambda=100`; the final strict dynamics block selected `lambda=10`, and the two auxiliary logits selected `lambda=0.0001`. For candidate 03, first-five dynamics selected `lambda=1`, centroid dynamics `lambda=100`, final strict dynamics `lambda=1`, and auxiliary logits `lambda=0.0001`.

The selected guarded-nonlinear configurations were:

| Candidate | Depth | Learning rate | Iterations | Minimum leaf | L2 | Calibration intercept | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| 02 | 2 | 0.03 | 100 | 128 | 10 | -1.454 | 0.572 |
| 03 | 3 | 0.03 | 100 | 128 | 1 | -1.590 | 0.502 |

The slopes near 0.5 show substantial calibration shrinkage. The nonlinear model's opposite performance across candidates argues against treating its candidate-03 gain as a general nonlinear mechanism without a fresh test.

## 8. Secondary post-break audit

| Candidate | Futures reaching a break | Matrices represented | Strict events after break | Model fitted? |
|---|---:|---:|---:|---|
| 02 | 36,655 | 79 | 1,190 | Yes |
| 03 | 38,748 | 78 | 1,292 | Yes |

These models were fitted for possible secondary confirmation scoring. Because the primary pilot stopped, they were never tested on untouched data and cannot support a post-break prediction claim. Likewise, relaxed first-five and centroid outcomes cannot rescue the failed strict-family selection.

## 9. What the registered result means

### Supported by this pilot

- The implementation and full replay are sound for this cohort.
- The strict event was observed with ample count and catalytic-matrix support.
- Four prespecified enhanced families had positive development cross-fitted gains in all four candidate/half cells.
- Direct ridge and hurdle both appear promising enough to justify a newly registered follow-up if the scientific value warrants the cost.
- The most likely source of selection failure is candidate-dependent model preference, not an absence of any predictive signal.

### Not supported by this pilot

- No single registered model family passed the 75% stability gate.
- No predictor is frozen for confirmation.
- The planned 200-matrix confirmation is not authorized under this protocol.
- Pilot gains are development estimates, not confirmation evidence.
- The pilot does not isolate composition, beta, interaction, or local dynamics as the mechanism.
- It does not demonstrate causal control, molecular intervention efficacy, recurrence, attractor switching, biological memory, or origin-of-life realism.

The correct paper-facing statement is therefore:

> A fresh 80-matrix pilot found consistent but model-unstable predictive improvement for the strict break-and-distinct-renewal event. The preregistered family-selection gate failed, so prediction beyond complete direct history/all clocks remains unconfirmed and no confirmation was run.

## 10. Explicitly post-hoc recipe comparisons

The following recipes were calculated only after observing the registered failure. They are decision aids, not pilot successes, and they have no registered bootstrap-selection frequency.

| Post-hoc recipe | Candidate-equal loss | 02 A gain | 02 B gain | 03 A gain | 03 B gain |
|---|---:|---:|---:|---:|---:|
| Equal probability mean: direct + hurdle | 0.110626 | 0.006403 | 0.005324 | 0.008694 | 0.009464 |
| Equal logit mean: direct + hurdle | 0.111211 | 0.006275 | 0.005248 | 0.007593 | 0.008429 |
| Equal probability mean of four eligible families | 0.111476 | 0.005900 | 0.005067 | 0.007488 | 0.008029 |
| Candidate-specific direct-02/hurdle-03 | 0.110238 | 0.006568 | 0.006000 | 0.009044 | 0.009827 |
| Candidate-specific direct-02/nonlinear-03 | **0.108733** | 0.006568 | 0.006000 | 0.012970 | 0.011921 |

The equal-probability direct+hurdle ensemble is the strongest conservative common recipe. It applies the same fixed averaging rule to both candidates, improves all four cells, and performs better in pilot cross-fitting than either common constituent alone. The candidate-specific nonlinear recipe has the lowest pilot loss but is the most outcome-adaptive and therefore the most vulnerable to selection optimism.

None of these calculations changes the registered `passed=false` result.

## 11. Options for the next step

### Option A — Stop and report the registered outcome

Treat the event's occurrence as established but its predictability as unresolved. This is the least expensive and strongest protection against researcher degrees of freedom. It leaves causal-control experiments deferred.

### Option B — New registered ensemble confirmation (recommended if prediction is the priority)

Create a versioned protocol that treats this pilot as development data and freezes, before any new simulation:

- Candidate-specific direct and hurdle models already fitted in `all_pilot_models.pkl`.
- The exact enhanced prediction `0.5 * direct_probability + 0.5 * hurdle_probability`.
- No recalibration, new tuning, family selection, or weight selection.
- A new seed domain and 200 untouched matrices with 128 futures per state and complete replay.
- The same four candidate/half log-loss, bootstrap-lower-bound, and Holm-randomization gates.
- Direct and hurdle constituent scores as secondary diagnostics only.

This tests a narrower new question: whether a prespecified ensemble transfers, not whether one mechanism was selected. Because the ensemble was invented after this pilot, the new registration and untouched cohort are mandatory.

### Option C — Candidate-specific frozen families

Freeze direct ridge for candidate 02 and hurdle for candidate 03, then test them on a new untouched cohort. Pilot performance is slightly better than the common ensemble, but the claim becomes simulator-specific and provides less evidence for a shared mechanism. The direct-02/nonlinear-03 combination is numerically strongest but carries still greater selection risk and is not the recommended first follow-up.

### Option D — Larger fresh selection pilot, then another confirmation

Use 200 new matrices to decide between direct, hurdle, and perhaps a fixed ensemble, followed by a second untouched 200-matrix confirmation. This is the strongest option if selecting a stable common architecture matters more than cost. It requires two new seed domains and substantially more simulation than Option B.

### Option E — Mechanism-first development study

Do not seek confirmation yet. Register a new development analysis that archives out-of-fold predictions for each hurdle component and explicitly tests candidate-by-feature-block interactions. This would ask whether candidate 02 is driven by direct state/network geometry while candidate 03 is driven by break/run8/strict stage structure. Any resulting recipe would still need later untouched confirmation.

### Option F — Proceed to molecular control

Not recommended. The predictor has not passed untouched confirmation, so intervention targeting would compound prediction uncertainty with causal uncertainty.

## 12. Recommendation

Do not alter the failed pilot or run its blocked confirmation command. If the objective is the quickest rigorous test of whether the present physical organization predicts the strict event, implement Option B as a new versioned experiment. The exact equal-probability direct+hurdle rule is simple, common across candidates, positive in every pilot half, and less outcome-adaptive than candidate-specific selection.

If the objective is instead to argue for a common dynamical mechanism, choose Option D or E. The present candidate split is too strong to interpret the direct, hurdle, nonlinear, or dynamics results mechanistically.

## 13. Source artifacts

| Artifact | Role |
|---|---|
| `../regime_prediction_pilot/PILOT_REPORT.md` | Original sealed minimal report; unchanged |
| `../regime_prediction_pilot/selection.json` | Registered selection result |
| `../regime_prediction_pilot/pilot_seal.json` | Experiment, power, replay, source, and stop seal |
| `../regime_prediction_pilot/replay_audit.json` | Exact replay evidence |
| `../regime_prediction_pilot/pilot_states.csv` | Cross-fitted state predictions and observed branch probabilities |
| `../regime_prediction_pilot/pilot_arrays.npz` | Complete feature, endpoint, stage, margin, and trajectory arrays |
| `../regime_prediction_pilot/pilot_branches.csv.gz` | Every future's branch-level outcome |
| `../regime_prediction_pilot/pilot_windows.csv.gz` | Every eligible eight-run window |
| `../regime_prediction_pilot/all_pilot_models.pkl` | All fitted candidate/family models; retained for audit |
| `../regime_prediction_registration/protocol.json` | Prospective design and decision rules |

The original sealed `PILOT_REPORT.md` SHA-256 is `c0d0118b28de719026464f8c0690c68c3c72cb8acc00e70659705421a4027437`. The sealed `selection.json` SHA-256 is `282c39859032c8733b7bef41404e53643189ad3dd52beeaf29c02ba2e5049902`. The sealed `pilot_seal.json` SHA-256 is `a07946e4cf2326ab73adc645b43139baaa4a3a3e950d8b3f626ece60ebd37ead`.
