# Input-Paper Figure Contents and Captions for Human Review — V2

## Top summary

- **Research step ID:** `S19-L13` (`E01-S19-L13-FIGURE5-RECURRING-TARGET-PREDICTION-RECONSTRUCTION-v1.0.0`), report-assembly addendum only.
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`.
- **Artifact written:** `FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md`, covering the supplied paper's Figures 1–6 and Table 1.
- **Validation result:** `PASS`; the paper PDF and all eight native extracted figure assets match the frozen SHA-256 identities listed below, and every panel in Figures 1–6 is represented.
- **Outcome classification:** L13 remains `FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED`; this V2 document only records my reading of the **input paper**, not a new scientific result.
- **Caveats or blockers:** Caption text below is a faithful paraphrase, not a verbatim transcription. Numerical readings marked “approximately” are visual estimates frozen in L12. Several paper-visible operations remain under-specified or internally inconsistent.
- **Recommended next action:** Compare this reading panel by panel with the supplied PDF/native images and flag any mismatch in what I think the paper shows or claims. Keep S20, E02, L14, confirmation, and interventions inactive.

## Scope and source identity

This V2 is the requested human-verification aid for the **paper's own figures and captions**. It is deliberately separate from V1, which documents L13-generated plots.

- Paper PDF: `/cache/e01_s03/downloads/paper-2607.28250v1.pdf`
- Paper PDF SHA-256: `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`
- Native extracted image directory: `/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures`
- Printed figure pages: 14–17 of the supplied preprint.

| Paper component | Native file(s) | Frozen SHA-256 |
|---|---|---|
| Figure 1 | `figure-01.png` | `d26f2de1bd79dfea4fdd12bc8cfc9b5ee4bbcfbf62e73928a2e792f643a49710` |
| Figure 2 | `figure-02.png` | `0e4aac507ccf6e10ced31edd6d7e5ba8c876d9d0c8d420b145dfc27c7d040778` |
| Figure 3 | `figure-03.png` | `7bd35a0b09a679d9b2f5c0fe8c57ea02b39833c663383bbdb676d2cbecf5c0c8` |
| Figure 4 | `figure-04.png` | `8632a9fe080d80066a9a5925e80c15aac4962393260709d7cebebdec617b224b` |
| Figure 5 | `figure-05.png` | `75be305d13203e65a8c93464b8a23aa25a86a567880458e889137bfe1281a968` |
| Figure 6A | `figure-06.png` | `42a542c10467e80aad139e772055a569689085e1d08cea8250636104a24dd498` |
| Figure 6B | `figure-07.png` | `a194e06ab0698f3b82f7eb2dee864644eb794a550d496a4f9fe1ee32d9aeb943` |
| Figure 6C | `figure-08.png` | `856678b09e71c4fbcc32db39a75fec13acbf1629c5eeb979b072826f1aa82e67` |

For each item below, **Visible content** is my direct reading of the native image; **Caption meaning** is my paraphrase of the supplied caption; **Operational reading** states the computation the panel appears to require; and **Ambiguities** records what cannot be learned by eye.

## Figure 1 — End-to-end conceptual system

**Source:** printed page 14; native `figure-01.png`.

### Panel A

- **Visible content:** A colored molecular assembly appears above a directed catalytic-reaction network. Colors denote molecule types; arrows denote catalysis among types.
- **Caption meaning:** GARD models compositional assemblies drawn from a fixed molecular vocabulary whose interactions are governed by a catalytic network.
- **Operational reading:** Each run needs a catalytic matrix/network and a molecular-count composition state.
- **Ambiguities:** The panel does not specify the catalytic-matrix distribution, direction/index convention, weight use, or initial-state sampling.

### Panel B

- **Visible content:** Environmental molecules accrete along “molecular time”; an assembly reaches a critical size, fissions into two daughters, one daughter continues, and the lineage repeats to 100 generations. The schematic itself says the molecular evolution uses ODE dynamics.
- **Caption meaning:** A selected lineage grows, fissions at a size boundary, and continues from one progeny until the generation limit.
- **Operational reading:** Molecular observations occur inside generations; fission marks a second clock; continuation selects one daughter.
- **Ambiguities:** The image does not say which daughter is selected, how overshoot/extinction is handled, or how molecular observations are recorded. Its “ODE dynamics” wording is visibly in tension with the Methods description of stochastic Poisson updates.

### Panel C

- **Visible content:** Similar compositions recur at separated points along the lineage and are enclosed as a tight composition-space cluster. Text in the panel equates tight clusters/homeostatic growth with attractors and self-replicators.
- **Caption meaning:** Self-replicators are recurring composition-space clusters with homeostatic, attractor-like behavior.
- **Operational reading:** Replicator status appears to require recurrence relative to a composition-space attractor, not merely similarity to the immediately previous molecular state.
- **Ambiguities:** The panel does not specify distance, threshold, clustering method, single-versus-multiple clusters, centroid/medoid, recurrence count, molecular-versus-boundary clock, retrospective fitting, or projection onto molecular time.

### Panel D

- **Visible content:** Several molecular compositions feed into one local `Φ^r` trajectory plotted over molecular steps; the sample trace is noisy with a late high excursion.
- **Caption meaning:** Relative composition at every molecular step is transformed into one `Φ^r` value, yielding a per-run molecular-time trajectory.
- **Operational reading:** The information pipeline consumes the molecular composition series, not only the 100 fission-boundary states.
- **Ambiguities:** The panel does not reveal preprocessing, partition, estimator, scalar identity, full-run versus prefix fitting, or whether the first lagged observation is dropped.

**Human checks for Figure 1:**

- [ ] Does panel C visually support a recurring-attractor label rather than adjacent-state smoothness?
- [ ] Is panel C singular (one dominant attractor) or plural (membership in any recurring cluster)?
- [ ] Does panel B explicitly imply one continuing daughter and two distinct clocks?
- [ ] Do you agree that the visible ODE wording conflicts with the stochastic-update Methods text?

## Figure 2 — Aggregate and individual `Φ^r` dynamics

**Source:** printed page 15; native `figure-02.png`.

### Panel A

- **Visible content:** A pale blue median-with-standard-deviation aggregate spans roughly 0–1,300 molecular steps and sits near zero overall. A red linear fit is nearly flat and is annotated `p=0.1995 > 0.05`. Large vertical dispersion/excursions appear at scattered and terminal positions.
- **Caption meaning:** Across 100 runs, the large-scale median trajectory has no significant linear trend.
- **Operational reading:** Unequal-length molecular trajectories were aligned somehow, summarized pointwise by median and standard deviation, and regressed over the displayed molecular index.
- **Ambiguities:** Padding, truncation, available-case tails, minimum contributor count, and the exact regression series are invisible. The x extent exceeds many sample-run lengths.

### Panel B

- **Visible content:** One run to about 800 steps, with a baseline near one, two positive rectangular plateaus near 8–10, and several abrupt negative drops near -60.

### Panel C

- **Visible content:** One run to about 800 steps, with multiple positive plateaus near 3–4 and many narrow or rectangular negative excursions reaching roughly -15.

### Panel D

- **Visible content:** One run to roughly 1,050 steps, nearly zero except for a very narrow paired positive/negative excursion around the middle (approximately +100 and -165).

- **Caption meaning for B–D:** Individual trajectories contain punctuated positive and negative spikes despite the absent aggregate trend.
- **Operational reading:** The plotted object is a signed local `Φ^r` value at molecular resolution; abrupt plateaus and paired extremes may encode partition changes or numerical conditioning as well as dynamics.
- **Ambiguities:** The caption does not define the three-standard-deviation scope, run-selection rule, completed-fit dependence, or numerical filtering. Panel A's “median ± std” also leaves contributor handling unresolved.

**Human checks for Figure 2:**

- [ ] Are the B/C excursions visibly rectangular as well as spike-like?
- [ ] Does D show both a very large positive and negative excursion at nearly the same time?
- [ ] Does A visibly extend to about 1,300 steps while B/C end near 800 and D near 1,050?
- [ ] Is the red aggregate fit visually near-flat with `p=0.1995`?

## Figure 3 — Runwise association between replication and `Φ^r`

**Source:** printed page 15; native `figure-03.png`.

### Panel A

- **Visible content:** A histogram of 100 runwise Spearman coefficients spans approximately -0.15 to +0.55. A zero reference is dashed dark/blue; a red dashed mean is labelled about `ρ=0.139`.
- **Caption meaning:** Runwise correlations are on average positive, with a one-sample diagnostic declaring the mean significantly above zero.
- **Operational reading:** Each run contributes one coefficient; molecular observations are not the inferential replicates for the population histogram.

### Panel B

- **Visible content:** Four category bars: positive/significant about 54%, positive/non-significant about 19%, negative/significant about 6%, and negative/non-significant about 21%.
- **Caption meaning:** Positive and significant association is the largest category and comprises 54 of 100 runs.
- **Ambiguities and visible text conflict:** The figure caption says the coefficient uses **changes in** `Φ^r`, while the Results prose describes correlation between the **level of** `Φ^r` and replication. These are distinct analyses. The significance threshold, multiplicity treatment, and precise binary label remain unspecified by the panels.

**Human checks for Figure 3:**

- [ ] Is the mean marker visibly labelled near 0.139?
- [ ] Do the four bars visually correspond to 54/19/6/21 percent?
- [ ] Do you agree that the caption says change while the Results paragraph says level?

## Figure 4 — Drift versus self-replicating state contrast

**Source:** printed page 16; native `figure-04.png`.

### Panel A

- **Visible content:** Approximately 100 colored paired lines connect each run's mean `Φ^r` in drift (left) to its mean in self-replication (right). Many rise, some fall, and a few drift-state values are strong negative outliers.
- **Caption meaning:** Within each run, the authors compare state-conditioned mean `Φ^r`; 57% of runs are said to be higher in the replicating state.
- **Operational reading:** Each trajectory must contain both a binary replicator label and local `Φ^r`; means are computed within state and joined within run.
- **Ambiguities:** It is not visible how runs missing either state are handled, whether “mean” follows any trimming, or whether the label is molecular or projected from boundaries.

### Panel B

- **Visible content:** An across-run median line rises from drift to self-replication, with broad shaded standard-deviation bands (roughly centered near 0.2 and 0.8, respectively).
- **Caption meaning:** The across-run summary is intended to show higher `Φ^r` in self-replicating states.
- **Ambiguities:** The caption refers to the median/standard deviation of “the lines in B,” which appears to be a panel-letter typo for A. The Results text mentions Mann–Whitney and Fisher combination, but panel aggregation alone does not reveal the exact test scope.

**Human checks for Figure 4:**

- [ ] Are both increasing and decreasing runwise lines visible despite the positive median shift?
- [ ] Are the large negative drift outliers visible in A?
- [ ] Does B show median ± standard deviation rather than a confidence interval?

## Figure 5 — First-quarter-to-final-three-quarter prediction

**Source:** printed page 16; native `figure-05.png`.

- **Visible content:** Five boxplots show binary accuracy for `Φ^r`, composition change, raw compositions, fluxes, and a majority dummy. Approximate centers are 0.85, 0.80, 0.79–0.80, 0.79, and 0.60. Significance brackets compare `Φ^r` with each alternative using two or three stars.
- **Caption meaning:** An MLP using the first 25% of a run's `Φ^r` is claimed to predict self-replicator appearance/state in the final 75% better than the same model using ordinary composition/flux features or a majority dummy; the worst reported comparison has `p<0.01`.
- **Operational reading:** The boxplots represent ten repeated matrix-level 80/20 train/test experiments. The target is a suffix binary sequence; the input is a first-quarter feature sequence.
- **Ambiguities:** The image does not expose tensor layout, masks, padding, balancing, validation split, scaling, target clock, “initial appearance” versus future occupancy, or whether `Φ^r` was fitted on the completed trajectory. The Results says each repetition is a dot/scatterplot, while the native panel is visibly rendered as boxplots without visible individual dots.
- **Cross-paper conflict:** A majority dummy near 0.60 implies a task-level majority prevalence near 0.60, whereas Table 1 reports control replication probability near 0.88. Those cannot be the same unbalanced molecular target and denominator unless an unreported sampling, balancing, label, or dataset transformation intervenes.

**Human checks for Figure 5:**

- [ ] Is the dummy center visibly near 60%, not 88% or 98%?
- [ ] Is `Φ^r` centered near 85%, with other learned families near 79–80%?
- [ ] Are boxplots, rather than ten visible dots, what the panel actually displays?
- [ ] Does the title/caption use “initial appearance” even though the described output is the full final 75% state sequence?

## Figure 6 — Intervention pipeline and treatment outcomes

**Source:** printed page 17; native `figure-06.png` (A), `figure-07.png` (B), and `figure-08.png` (C).

### Panel A

- **Visible content:** A loop begins just after fission, enumerates adding or deleting one molecule of each illustrated type, chooses the action that maximizes or minimizes `Φ^r`, simulates one GARD generation, and repeats.
- **Caption meaning:** Intervention occurs immediately after every fission; the selected single-molecule edit is the raw score extremum, followed by ordinary dynamics until the next generation.
- **Operational reading:** The scorer must assign one `Φ^r` value to every hypothetical edited post-fission state using only the information intended to be available at that decision.
- **Ambiguities:** No-op handling, refitting, partition/statistics reuse, future data, tie-breaking, numerical separability, and random-action controls are invisible. The schematic's small color set is illustrative rather than the stated 100 molecular types.

### Panel B

- **Visible content:** Persistence boxplots are ordered max > control/base > min, with centers near 874, 716, and 559 molecular steps and three pairwise significance brackets marked with three stars.
- **Caption meaning:** Maximizing `Φ^r` is claimed to increase self-replication persistence, while minimizing it decreases persistence; the caption attributes significance to Mann–Whitney tests.
- **Operational reading:** Persistence is the per-trajectory sum of positive molecular replicator labels, compared across treatment matrices.

### Panel C

- **Visible content:** Over 0–100 generations, max (blue) rises approximately 86%→89%, control (orange) is nearly flat around 88%, and min (green) falls approximately 81%→79%; shaded bands are labelled 95% confidence intervals. Legend annotations show slopes/statistics about +0.041 (`p<0.001`), +0.008 (`p=0.4659`), and -0.030 (`p=0.0034`).
- **Caption meaning:** Repeated maximizing interventions are claimed to accumulate a positive effect on replication probability, while minimizing interventions have the opposite trend.
- **Operational reading:** Molecular replication labels must be aggregated within generation (or another generation-indexed window), then treatment-level regressions and intervals are computed over generation.
- **Ambiguities:** The panel does not reveal the regression unit, within-generation denominator, repeated-measures treatment, contributor count by generation, or relation between curve averages and Table 1's overall means.

**Human checks for Figure 6:**

- [ ] Does A show intervention immediately after fission and exactly one selected add/delete action?
- [ ] Does B visibly order max > base/control > min in persistence?
- [ ] Does C show max rising, control nearly flat, and min falling with the stated approximate slope annotations?
- [ ] Is there any visible random-action arm or action-score uncertainty? My reading is no.

## Table 1 — Paper-facing outcome values and note

**Source:** printed pages 17–18, directly beneath Figure 6.

| Treatment | Persistence | Probability | Consistency | Time to first replicator |
|---|---:|---:|---:|---:|
| max `Φ^r` | 874 ± 233 | 88 ± 3% | 0.52 ± 0.04 | 36 ± 26% |
| control | 716 ± 198 | 88 ± 3% | 0.38 ± 0.06 | 37 ± 27% |
| min `Φ^r` | 559 ± 99 | 80 ± 3% | 0.42 ± 0.04 | 40 ± 28% |

- **Caption/note meaning:** Persistence is total positive molecular steps; probability is the positive-step fraction; consistency is Pearson correlation of consecutive labels; time to first is described in the note as molecular steps.
- **Unresolved dispersion:** The table does not identify whether `±` denotes SD or SE.
- **Internal conflict 1:** The first-onset cells print percent signs, but the note defines molecular-step counts.
- **Internal conflict 2:** The prose says minimization worsened all four properties and that higher consistency is better, yet min consistency (0.42) exceeds control (0.38).
- **Internal conflict 3:** Max and control both round to 88% overall probability even though Figure 6C shows visibly different time trends. This can be arithmetically possible, but the aggregation/window definition is missing.
- **Cross-figure conflict:** The 88% control probability conflicts with Figure 5's approximately 60% dummy if the same unbalanced molecular label and denominator were used.

**Human checks for Table 1:**

- [ ] Are the four values and dispersions transcribed correctly for all three treatments?
- [ ] Do the first-onset values visibly carry percent signs?
- [ ] Does the following note nevertheless define first onset in molecular steps?
- [ ] Do you agree that min consistency is numerically above control despite the “worsened” wording?

## Cross-figure interpretation I would use unless you correct it

1. **Replicator object:** Figure 1C depicts recurrence around one or more composition-space attractors. It does not visually justify the adjacent molecular `H>0.9` label used in the original frozen S13Y comparator.
2. **Information clock:** Figures 1D and 2 use molecular-step `Φ^r`; Figure 6 decisions occur at generation/fission boundaries. A coherent intervention implementation therefore needs an explicit mapping from one hypothetical boundary edit to a molecular-history information score.
3. **Level versus change:** Figure 3's caption and Results prose specify different estimands. I would preserve both rather than silently choose one.
4. **Prediction target:** Figure 5's 60% dummy is consistent with a roughly 40/60 target geometry, but not with Table 1's 88% positive molecular occupancy absent an unreported transformation.
5. **Completed-fit dependence:** Nothing visible in Figures 1–5 establishes that first-quarter `Φ^r` was fitted without the final 75%; public PhiRL behavior makes this a material ambiguity.
6. **Intervention semantics:** Figure 6 visually specifies timing and raw max/min action intent, but not the online scorer, refit scope, ties, numerical uncertainty, or matched/random outcome controls.
7. **Single coherent pipeline:** The paper figures do not, by themselves, identify one end-to-end implementation consistent with every panel and Table 1. This remains an author-code discrimination problem, not a reason to rewrite prior evidence.

## Requested human-review boundary

Please use the checkboxes above to identify any place where my visual reading differs from yours. A correction to what is visibly present should update this report description only; it must not retroactively change frozen scientific results. No next scientific loop is activated by this artifact.
