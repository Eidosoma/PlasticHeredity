# S19-L12 full results — Paper–PhiRL forensic concordance audit

## Concise top summary

- **Research step ID:** `S19-L12` — `E01-S19-L12-PAPER-PHIRL-FORENSIC-CONCORDANCE-AUDIT-v1.0.0`.
- **Completion status:** COMPLETE; analysis-only audit frozen at the mandatory human-review boundary.
- **Artifacts written:** 40 required named records, 12 required figures, source/concordance locks, and root S19 handoff/ledger updates.
- **Validation result:** PASS_ALL_REGISTERED_IMMUTABILITY_SOURCE_COVERAGE_PROHIBITION_REGENERATION_STORAGE_AND_HASH_GATES Source identity, immutable priors, all figure/table panels, all required PhiRL functions, all 59 S18 claims, all seven S18 prospective/causal questions, and S19-L01–L11R are cross-referenced. No GARD outcome, matrix, label, MLP, intervention, or new metric was produced.
- **Outcome classification:** `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION` (constraining/contradictory forensic result); secondary finding `PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT`.
- **Caveats or blockers:** The manuscript-specific code/data are unavailable; native-figure measurements are approximate visual constraints; public PhiRL is lineage evidence, not proof of the authors' hidden pipeline.
- **Recommended next action:** Mandatory human review. Keep S20 inactive. Recommended scientific posture: `AUTHOR_CODE_WAIT_STATE`; if authoritative implementation later appears, freeze it and run one untouched end-to-end reconstruction.

## Lay summary

We are closer to knowing *why* exact replication remains elusive, but not closer to claiming that a hidden author implementation has been found. The public PhiRL code gives a precise information-calculation pipeline, yet the paper's own equation, its “one atom” description, and PhiRL's two named outputs do not describe the same mathematical quantity. The paper also shows a roughly 60% majority baseline in its prediction figure while reporting roughly 88% replication in Table 1; those cannot be the same unbalanced molecular target under one denominator. Finally, the paper does not publish the GARD label, prediction tensor, or one-state intervention scorer needed to reproduce the later figures.

One public-source pipeline does explain parts of the paper: completed-fit PhiRL trajectories have punctuated spikes and retrospective label-coupled associations. But the strongest frozen E01 tests show aggregate-trend disagreement, future dependence, a deterministic ~98% adjacent-H target, no prospective feature advantage, and no max-intervention benefit. Repeated source-grounded label reconstructions did not recover the full occupancy/onset/consistency fingerprint. The most defensible conclusion is therefore not “the paper is wrong,” and not “we found its code,” but that public evidence admits incompatible pipelines and author implementation is required to discriminate them.

## Frozen question and scope

L12 asked what complete operation chain is required for Figures 1–6 and Table 1, and whether one public, source-grounded pipeline explains those visible fingerprints together with frozen E01 evidence. It was explicitly an audit, not a new scientific experiment. All S01–S18 totals remain unchanged: {'NOT_SUPPORTED_WITHIN_TESTED_SCOPE': 21, 'DIRECTIONALLY_SUPPORTED': 17, 'NOT_EVALUATED': 16, 'SUPPORTED': 3, 'UNDERDETERMINED': 2}. L11R remains `ALL_COMPTYPE_UNION_NOT_SUPPORTED`, `SOURCE_TAG_SINGLETON_DEPENDENT`, `NOT_PROMOTABLE`.

## Inputs and provenance

- Original arXiv preprint 2607.28250v1 PDF, SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`, plus workspace Markdown and eight native-resolution extracted images.
- PhiRL repository at pinned/current master `a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; current master and pinned tree are byte-identical.
- IIGR `7c1c22fe39f539d4a453135476f1f0dd5a6b45f7`, historical GARD `86dff6320d5ae91b4e831471079ff46749b14df9`, and BreakingGRNMemories `afe44231ad3ce915172cdb53a6b234bd76fcb6a5` as source-lineage context.
- Safe 16-node PhiID lattice JSON from S12B. The raw pickle was opened only by an isolated `python -I` restricted-conversion subprocess and compared with the frozen JSON.
- Every frozen S18 Matrix A/Matrix B row and all S19 loop classifications through L11R.

The complete file-level source and prior hashes are in `source_snapshot_manifest.json`, `phirl_repository_tree.json`, and `immutable_prior_validation.json`. Cached source without a compatible detected license is referenced by identity and hash, not redistributed.

## Methods and commands

The preregistered order was enforced: (1) freeze source/prior identities; (2) build sentence, panel, function, atom and leakage registries; (3) freeze their concordance hash; (4) only then generate and rank at most three whole-pipeline hypotheses; (5) design but do not execute one next action; (6) regenerate, validate and hash artifacts.

Primary commands:

```text
python scripts/e01/run_s19_l12.py prepare
python scripts/e01/run_s19_l12.py audit
python scripts/e01/run_s19_l12.py finalize
python -m pytest -q tests/e01/test_s19_l12.py
python -m compileall -q scripts/e01/run_s19_l12.py tests/e01/test_s19_l12.py
```

Execution used CPU float64 where numerical fixtures were needed, one process for audit logic, zero GPU, no simulator import, and no scientific random outcome. Graph layouts use fixed seeds. Figure digitization is recorded as approximate or reported-exact in `figure_digitization.csv`; it is never treated as raw author data.

## Paper sentence and dependency audit

`paper_statement_registry.parquet` contains 96 computationally meaningful rows: 37 method/semantics statements plus all 59 claim-ledger statements. Each row records objects, clock, unit, preprocessing, estimator, label, denominator, aggregation, test, reported value, specification state, E01 crosswalk and unresolved fields. The dependency graph shows that one label definition affects Figures 3–6 and Table 1, while one metric/scorer chain affects Figures 2–6.

The audit records 16 material discrepancies. The strongest are:

1. Figure 3 Results uses level while its caption specifies change.
2. Figure 5's ~60% dummy baseline conflicts with Table 1's 88% molecular occupancy if label and denominator are shared.
3. The displayed Phi-r equation, “one atom” prose, public `integrated`, and public `emergence` are non-identical.
4. Table 1 min consistency (.42) exceeds control (.38) despite “worsening all four” prose.
5. First-onset cells contain percent signs while the note defines molecular steps.
6. Figure 2 unequal-length aggregation and Figure 6 hypothetical-state scoring are not specified.

![Paper dependency graph](figure_01_paper_dependency_graph.png)

*Figure 1. Every downstream panel depends on unresolved choices in the label and/or scalar pipeline.*

## Figure-by-figure results

All 20 required Figure 1–6/Table 1 components are present in `figure_panel_registry.parquet`.

### Figure 1

Figure 1 jointly implies molecular-time Phi values and recurrence/attractor semantics across generations, but it does not state whether the binary label is one cluster, any cluster, a full-run reference, or a projected boundary state. L02–L11R explored these structural interpretations without finding a joint paper fingerprint.

### Figure 2

The aggregate extends to roughly 1,300 molecular steps while frozen compatible trajectories vary from roughly 200 to 1,467 observations. Frozen support falls sharply in the tail, so padding, available-case calculation, truncation, or resampling can materially change the trend. S14 reproduces punctuated excursions and temporal dependence but finds significant positive aggregate trends in both candidates, rather than p=.1995 trendlessness. Completed-fit/prefix values differ, and spikes can track partition changes and numerical condition.

![Figure 2 constraints](figure_04_figure2_digitized_clock_constraints.png)

*Figure 2. Native-figure x extents (left) and frozen trajectory support by molecular index (right).*

### Figures 3 and 4

The paper's LEVEL and CHANGE descriptions are internally inconsistent, so both remain named. Frozen S15 values are directionally positive in both candidates and paper-like one-sample diagnostics can be reconstructed, but these are retrospective, label-coupled, completed-fit results. Exact H completely determines frozen Y, ordinary stability is coupled, and past-only refitting reverses the direction. Figure 4 test scope and Fisher inputs remain missing.

![Figure 3 inconsistency](figure_05_figure3_level_change_inconsistency.png)

*Figure 3. Both frozen analyses remain below the paper's mean but point positively; neither replaces the other.*

### Figure 5

The paper's visible majority baseline is approximately .60. Table 1 reports .88 control occupancy, while S16's valid test suffix prevalence is approximately .983–.985. `figure5_reconciliation_possibilities.csv` enumerates class/per-run balancing, onset-only targets, molecular/generation targets, padding, masking, common-length truncation, negative enrichment, stratification, separate data and different labels. None is both specified by the paper and implemented by public PhiRL. S16's one prospectively frozen masked molecular layout does not reproduce the paper: dummy and learned models track prevalence, balanced accuracy is near .5, all test runs are already positive by the cutoff, and PhiRL adds no performance beyond H/stability controls.

![Figure 5 contradiction](figure_06_figure5_prevalence_contradiction.png)

*Figure 4. The prediction task's visible baseline cannot be the frozen unbalanced molecular target under a shared denominator.*

### Figure 6 and Table 1

The action timing and exhaustive add/delete idea are stated, but the mapping from one hypothetical state to a trajectory-fit Phi-r score is absent. S17's append-and-refit-current-prefix scorer is online and exactly replayable; its max arm does not improve over control and its min arm only modestly harms outcomes. It therefore constrains one coherent implementation but cannot adjudicate an unpublished scorer. Table units/dispersion and min consistency remain internally conflicted.

![Figure 6 and Table 1 map](figure_07_figure6_table1_consistency_map.png)

*Figure 5. Reported fields and the principal internal/frozen inconsistencies.*

## PhiRL executable source audit

All 10 required functions are located, blamed and traced. Public current master equals the pinned commit, so current-versus-pinned drift does not explain discrepancies. Internal history does matter: early PhiRL used the slow forward+backward-MI sum and `local_phi_r`; later commits exposed `emergence=synergy+causation`, then switched main execution to the fast averaged-correlation MI and trace-scaled covariance regularizer.

The current chain filters inactive variables and z-scores them using the complete trajectory; computes fast lagged correlations; with `alpha=1, bonferonni=False` retains every finite edge with p<1; adds a 1e-6 graph floor; takes an unnormalized Fiedler sign split (exact zero coordinates are omitted); averages each partition arithmetically; fits Gaussian means/covariances globally with a trace-scaled regularizer; inverts all 16 local PhiID atoms; and exports both nine-atom `integrated` and three-atom `emergence`. The shuffled path permutes the whole trajectory. `_load_phi` maps infinities to NaN, takes finite medians and can leave errors/missing cells as zeros.

![PhiRL flow](figure_02_phirl_dataflow_graph.png)

*Figure 6. Public PhiRL's executable chain; almost every fitted prefix value inherits full-trajectory dependence.*

![Future dependence](figure_09_completed_fit_future_dependence.png)

*Figure 7. Red operations use or inherit the completed future suffix.*

No public branch, tag, deleted path or inspected lineage file contains the GARD label, Figure 5 tensor, GARD MLP, hypothetical-state intervention scorer, treatment simulations, or Table 1 aggregation.

## Metric-identity result

All 16 atoms were recovered from the safe lattice. If source antichains are redundancy `r`, uniques `u0/u1`, and synergy `s`, the displayed equation expands as `Σ_q(s→q-r→q)`: eight signed atoms. Public `emergence` is only `s→s+s→u0+s→u1`. Public corrected `local_phi_r` contains nine different integrated atoms. These coefficient patterns are not algebraically equal.

![Metric atom map](figure_08_metric_identity_atom_map.png)

*Figure 8. Public integrated, public emergence, and the displayed paper equation are distinct atom combinations.*

The audit therefore classifies metric identity as `PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT`, not as any one favorable scalar. This does not invalidate a private coherent implementation; it means the public record does not identify it.

## Paper–PhiRL–E01 concordance

The concordance matrix has 112 rows: 33 material pipeline elements, 59 immutable S18 claims, seven immutable S18 Matrix B questions, and 13 S19 loop results. Prior statuses are copied, never rewritten.

![Concordance heatmap](figure_10_paper_phirl_e01_concordance_heatmap.png)

*Figure 9. Direct specification/source/result support and unresolved author dependence by material operation.*

The recurring pattern is not a single numeric miss. Direct paper support is strongest for broad concepts; direct public support is strongest for the information component; direct frozen evidence tests plausible joins between them; and the interfaces—label, aggregation, prediction tensor and intervention scorer—remain unpublished.

## Root causes and whole-pipeline hypotheses

The highest-leverage causes are the label/task mismatch, scalar identity mismatch, Figure 5 task/denominator mismatch, completed-fit dependence and the intervention scorer. Scores rank forensic informativeness, not closeness to desired outcomes.

![Root causes](figure_11_root_cause_ranking.png)

*Figure 10. Audit-priority ranking before any new scientific execution.*

After the concordance matrix was hash-frozen, exactly three complete hypotheses were retained:

- `HP1_PUBLIC_PHIRL_COMPLETED_FIT`: forensic score 20; Most directly public-source grounded and already strongly tested; reproduces some retrospective resemblance but fails label prevalence, prospective prediction and intervention ordering.
- `HP2_PAPER_LITERAL_WMS_RECURRING_ATTRACTOR`: forensic score 11; Closest to manuscript wording, but key operations needed to make it executable are absent from public code; prior recurring-attractor reconstructions did not recover the joint fingerprint.
- `HP3_FIGURE_SPECIFIC_MIXED_PIPELINES`: forensic score -5; Can narratively absorb the manuscript contradictions but lacks coherent public support and is too assumption-rich to execute as confirmation.

None is one coherent publicly supported paper pipeline. HP1 is most executable and most thoroughly falsified as a full-paper reconstruction. HP2 is closest to wording but cannot be executed without identity-changing inventions. HP3 explains contradictions by allowing figure-specific pipelines, but that flexibility makes it poorly source-grounded and weakly falsifiable.

## Classification and decisive next step

The registered audit classification is `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION`. This is stronger than merely “underdetermined” because exact public source inspection establishes that necessary GARD-specific operations are absent and manuscript-visible constraints conflict. It is not a scientific non-replication verdict beyond the frozen scope, and it does not modify S18.

![Decision tree](figure_12_decisive_next_step_tree.png)

*Figure 11. A new reconstruction cannot become confirmatory while its defining operations must be invented.*

The recommended next action is an author-code wait state, without contacting authors unless separately authorized. If exact code/configuration becomes available, the smallest defensible scientific action is one untouched, seed-firewalled, end-to-end Figure 1–6/Table 1 reconstruction that reports paper-facing, prospective and causal gates separately.

## Validation

- Immutable prior baseline: every scoped S01–S18, V1/V2 and S19-L01–L11R file passed SHA-256/size validation before and after the audit.
- Source freeze: current PhiRL master, local checkout and pinned commit/tree are identical; relevant source files and histories are hashed.
- Safe lattice: isolated restricted conversion matches the frozen JSON in raw hash, 16 nodes, edges, order and contents.
- Coverage: all required figure/table components and PhiRL functions are present; all 59+7 S18 rows and 13 prior S19 loop results are cross-referenced.
- Prohibitions: zero new GARD trajectories, matrices, scientific labels, emergence branches, MLPs or interventions; zero GPU use; no author contact.
- Ordering: `concordance_lock.json` predates and gates generation of candidate hidden-pipeline hypotheses.
- Regeneration: substantive tables, reports and all 12 figures are deterministically regenerated and compared in `regeneration_validation.json`.
- Artifact integrity: complete hashes and storage accounting are in `artifact_manifest.json` and `storage_validation.json`.

## Caveats, blockers and limitations

1. Native-figure digitization is deliberately approximate and cannot replace underlying data.
2. Git history establishes public chronology, not the private analysis date or unpublished code identity.
3. The safe-lattice algebra uses the public two-source/two-target convention; another redundancy convention could change numerical atoms but not reconcile the public coefficient definitions as written.
4. S19 explored many labels adaptively. Those results constrain interpretations but are not confirmation.
5. The Figure 5 contradiction admits multiple transformations; L12 does not select one by target proximity.
6. A coherent unpublished implementation may exist. Only its authoritative code/config/data can distinguish that possibility from manuscript inconsistency.

## Provenance and dependency versions

Python 3.13.14; NumPy 2.4.6; pandas 2.3.3; SciPy 1.18.0; scikit-learn 1.9.0; NetworkX 3.6.1; Matplotlib 3.11.1; CPU float64 authoritative; GPU unused. Repository implementation and preregistration were committed and pushed on `eidosoma/groups/42` before final audit release. Full source/blob hashes and commands are machine-readable.

## Mandatory boundary

L12 is complete. S20 and E02 remain inactive. No recommendation has been executed. Control returns for human review.
