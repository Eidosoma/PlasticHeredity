# Cluster Report: Paper Reconstruction

## Cluster scope

`PAPERRECON` covers the public-source integration and the direct reconstruction of the manuscript's self-replicator, Phi-r, prediction, and intervention claims: S12 through S18, the S19 umbrella, and S19-L01 through L17. It asks two separate questions:

1. How closely can the visible paper outputs be reconstructed from the manuscript and public source lineage?
2. Do the reconstructed methods support prospective prediction or causal control?

Those questions were never collapsed. The cluster ends with the immutable S18 verdict and the S20-preserved position: **partial directional retrospective reconstruction**, **prospective prediction not supported within tested scope**, **prospective causal control not supported within tested scope**, and **HOLD** for the original full-plan gate.

## Source and reconstruction methods

The analysis pinned the manuscript PDF and extracted figures, historical GARD source, PhiRL/IIGR lineages, PhiID lattice, software environments, and code commits. S12-series repairs were additive and fail-closed: every failed attempt and numerical or schema issue remained in the record. The final clean S13Y cohort used 100 shared catalytic matrices and 200 candidate-2/candidate-3 trajectories. Paper-facing stages then reconstructed:

- Figure 2: local information trajectories, aggregate trend, spikes, and temporal dependence;
- Figures 3–4: runwise correlation and replicator/nonreplicator contrasts;
- Figure 5: first-quarter feature tensors predicting the remaining three quarters;
- Figure 6 and Table 1: an online append-edit-refit intervention procedure;
- alternative self-replicator labels, exposures, clustering semantics, padding conventions, and related-team Phi pipelines.

All completed-fit Phi results were explicitly marked future-dependent. Candidate simulators remained separate.

## Paper-facing results

### Figure 2: punctuated excursions reproduced, aggregate trend did not

Both simulator candidates generated abrupt positive and negative information excursions and substantial temporal dependence. Positive three-standard-deviation excursions occurred in 90/100 runs per candidate, and differenced trajectories rejected temporal independence in 100/100. However, the paper reported no significant aggregate trend (`p=0.1995`), whereas the reconstruction produced significant positive slopes (`p=3.19e-13` and `p=4.28e-4`) under the preregistered available-case alignment. Molecular trajectories were unequal in length, so late aggregate indices had fewer contributing runs. Full-support, majority-support, and normalized-lifetime alternatives were retained but could not identify the manuscript's unpublished alignment; its Ljung–Box lag is likewise unspecified.

![Candidate-specific reconstruction of aggregate and individual Figure 2-like PhiRL trajectories.](figures/paper_s14_figure2_reconstruction.png)

*Figure PAPERRECON-1. Figure 2 reconstruction from S14. The lower panels show punctuated run-level excursions resembling the manuscript; the available-case aggregate panels reveal positive fitted trends rather than the reported null trend. Unequal tail support and unresolved author alignment/lag semantics prevent exact numerical equivalence from being adjudicated.*

### Figures 3–4: directional, retrospective resemblance

Runwise level and change correlations were positive in 75–81 of 100 runs depending on candidate and formulation, compared with 73 reported. Significant positive counts were 44–55, close to the reported 54 for some branches. Replicator-labelled states usually had larger completed-fit Phi values, and pooled or run-summary tests produced the reported direction. Yet the effect sizes and counts differed, and the strongest resemblance depended on a completed trajectory and a label exactly coupled to compositional similarity. The past-only direction reversed in key branches.

The correct interpretation is association under a reconstructed retrospective pipeline, not a signal available before the event.

### Figure 5: the task was not reconstructed as scientific prediction

Under the frozen adjacent-H label, real molecular observations were positive about 98% of the time. This produced a majority dummy near 98%, whereas the visible paper dummy was about 60%. All scientific balanced accuracies were approximately 0.5 and no matrix was still genuinely pre-onset at the first-quarter cutoff.

![Figure 5 prediction reconstruction across feature families and cutoffs.](figures/paper_s16_figure5_reconstruction.png)

*Figure PAPERRECON-2. Figure 5 reconstruction from S16. Raw accuracy is saturated by the nearly always-positive adjacent-H target, including the dummy. The plot demonstrates why a high nominal accuracy is not evidence of discrimination or prediction before first appearance.*

S19-L13 found that a recurring-attractor target with about 40% occupancy could naturally recover a roughly 60% dummy, but the paper's model ordering did not follow. L14–L15 showed that unmasked padding and length cues could produce very high all-cell accuracy, while masked valid-cell models remained nondiscriminative. L16 found no complete source-supported tensor/architecture hypothesis that reconciled all panels. Thus neither target prevalence nor padding alone reconstructed a scientifically valid Figure 5 pipeline.

### Figure 6 and Table 1: procedure executable, causal ordering unsupported

S17 exactly implemented a literal online procedure: at each selected post-fission boundary, enumerate molecular additions and present-molecule deletions; append each hypothetical edit; refit the current PhiRL prefix; choose raw maximum, no action, or raw minimum; and continue with common random streams until divergence. Across 72 treatment trajectories and 558,380 candidate-action ledger rows, the implementation replayed exactly.

![Literal online intervention pipeline, persistence, and trajectory-level treatment outcomes.](figures/paper_s17_figure6_reconstruction.png)

*Figure PAPERRECON-3. Figure 6/Table 1 reconstruction from S17. The left panel records the locked online scoring chain. Persistence distributions and generation-wise replication probabilities do not show the manuscript's required max ≥ control ≥ min ordering; all treatments rapidly approach the permissive adjacent-H positive state.*

The paper-facing ordering failed. Mean max persistence and occupancy were below control in both candidates; max-minus-control confidence intervals included zero; min effects were mixed; and action-separation gates failed. The result proves that one literal procedure is executable and reproducible, not that Phi controls replication.

## S18 claim-level verdict

The immutable 59-claim Matrix A contained:

| Status | Claims |
|---|---:|
| Supported | 3 |
| Directionally supported | 17 |
| Not supported within tested scope | 21 |
| Underdetermined | 2 |
| Not evaluated | 16 |

The supported items were largely technical or broad directional properties—for example, significant positive aggregate association diagnostics, differenced temporal dependence, online scoring execution, and intervention replay. The principal scientific claims of prospective prediction and bidirectional causal control did not pass.

## The 88% versus 98% and label forensics

S19-L01–L11R systematically tested adjacent smoothness, direct similarity, dominant compotypes, cross-generation recurrence, boundary recurrence, exposure changes, MATLAB-compatible clustering, and all-compotype unions. Results occupied incompatible ends of the paper fingerprint:

- adjacent molecular H > 0.9 was too permissive and early, with approximately 98% occupancy;
- dominant recurring-attractor labels were too sparse, late, and sticky, with molecular occupancies around 0.27–0.40;
- some settings reproduced occupancy near 0.88, but not persistence, onset, consistency, episode structure, and cross-candidate behavior jointly;
- an untouched L08 comparison concluded that neither the boundary mechanism nor high exposure reproduced the complete fingerprint;
- union labels were not supported and one historical source-tag branch depended on singleton clusters.

L12 then audited every paper panel, Table 1, and public PhiRL operation. It concluded that author code was required to discriminate complete hidden pipelines and that the paper's metric identity was internally inconsistent. In particular, Figure 5's roughly 60% dummy and Table 1's roughly 88% probability could not be reconciled as one clearly specified target/denominator.

L17 audited the related-team `BreakingGRNMemories` Phi lineage as inspiration. No unchanged transferred scalar produced a positive, independent result in both GARD candidates. Completed-fit variants remained retrospective and closely tracked stability.

## Contradictions and nulls

- Aggregate trend direction contradicted Figure 2's reported null trend.
- Completed-fit Phi association did not survive as a past-only direction.
- The Figure 5 dummy and target prevalence were incompatible under the adjacent-H label.
- Learned models did not beat ordinary controls on valid cells.
- Padding could create an artificial task but did not reproduce a valid scientific panel.
- Max/control/min intervention ordering was not recovered.
- The paper's most visible self-replicator fingerprints could not be explained by one registered label, clock, exposure, denominator, and clustering definition.
- Public source lineage did not uniquely determine the paper's Phi scalar or end-to-end GARD pipeline.

## Evidence assessment

| Manuscript component | Reconstruction degree | Scientific interpretation |
|---|---|---|
| GARD simulator family | High for explicit candidate branches | Reconstructed family, not author identity |
| Figure 2 excursions | Directionally close | Aggregate trend materially different |
| Figures 3–4 associations | Directionally close under completed fit | Retrospective and label-coupled |
| Figure 5 prediction | Not reproduced | No prospective first-appearance signal in tested scope |
| Figure 6 procedure | Mechanically reconstructed | Outcome ordering and causal control unsupported |
| Table 1 | Mixed, with major label/denominator conflict | Author implementation required |

## Cluster conclusion

The forensic program recovered enough of the manuscript's computational neighborhood to explain why some outputs look similar. It did not recover one coherent public pipeline capable of producing all figures and Table 1, and it did not validate PhiID as a prospective warning signal or control knob. This is a constraining result, not a failed project: it establishes which similarities are retrospective or label-induced and motivates the later search for a process outcome that can be defined and confirmed without privileged future information.
