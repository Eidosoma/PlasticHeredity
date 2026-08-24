# S19-L15 Full Results — Untouched Padding/Length Panel Discrimination

## Top summary

- **Research step:** `E01-S19-L15-UNTOUCHED-PADDING-LENGTH-PANEL-DISCRIMINATION-v1.0.0`
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Outcome classification:** `EXPLORATORY_NON_SUPPORT`; `EXPLORATORY_NON_SUPPORT, POSSIBLE_PIPELINE_ARTIFACT, NOT_PROMOTABLE`
- **Artifacts written:** the full registered machine-readable tables, 13 figures, immutable/source/seed/runtime/storage/regeneration manifests, this report, and the S19 handoff.
- **Validation:** all fixtures passed; 400/400 untouched trajectories and 400/400 complete per-trajectory feature payloads regenerated exactly; 24/24 registered model replays passed; prior artifacts remained immutable.
- **Central boundary:** all-cell padding resemblance is forensic only. Padding cells are not molecular observations, completed-fit PhiRL is future-dependent, and S18 prospective prediction and causal-control conclusions are unchanged.
- **Recommended next action:** mandatory human review; do not activate L16, S20, E02, author contact, intervention work, or report generation automatically.

## Frozen question and why the larger run was warranted

L14 established exact tensor replay but stopped before any MLP because its 100-matrix candidate-specific padding arithmetic straddled the digitized dummy interval. It also showed that first-quarter length almost exactly identifies the padded suffix boundary. L15 therefore used a prospectively frozen, new 200-matrix paired cohort to test the previously unopened mechanism: whether the exact S16 MLP, trained and scored over ordinary zero padding, produces the complete Figure 5 panel. The larger scope was set before outcomes to distinguish a stable mechanism from ten-split noise.

## Immutable methods

Candidate 2 used h=0.6031526490073492 and first-daughter continuation. Candidate 3 used h=0.5613315384859516 and random-nonempty continuation. Both used the frozen overshoot rule and 100-fission selected molecular clock. The sole target was strict adjacent-incoming `H>0.9`. The exact S16 288,789-parameter CPU-float64 MLP, quarter cutoff, right-zero padding, fixed feature construction, ten 128/32/40 matrix splits, and paper digitization were locked and pushed before any outcome. No threshold, model, feature, candidate, or digitization was selected from the result.

## Cohort and padding geometry

- CANDIDATE_2: 199/200 eligible; q=0.5883; real-label prevalence=0.9829; padded prevalence=0.5782; padded dummy=0.5782.
- CANDIDATE_3: 199/200 eligible; q=0.6200; real-label prevalence=0.9837; padded prevalence=0.6099; padded dummy=0.6099.

The algebraic identity `p_padded = p_valid × q_valid` passed. Target zeros beyond the valid suffix were made fully visible in every all-cell result and were never described as physical states.

## Paper-facing all-cell panel

- CANDIDATE_2: P1_PHIRL_EMERGENCE_COMPLETED_FIT=0.9786 (paper≈0.8485); B1_COMPOSITION_CHANGE=0.9786 (paper≈0.8054); B2_RAW_COMPOSITIONS=0.9796 (paper≈0.7992); B3_MOLECULAR_FLUXES=0.9810 (paper≈0.7900); D0_MAJORITY_DUMMY=0.5836 (paper≈0.6092).
- CANDIDATE_3: P1_PHIRL_EMERGENCE_COMPLETED_FIT=0.9784 (paper≈0.8485); B1_COMPOSITION_CHANGE=0.9782 (paper≈0.8054); B2_RAW_COMPOSITIONS=0.9793 (paper≈0.7992); B3_MOLECULAR_FLUXES=0.9797 (paper≈0.7900); D0_MAJORITY_DUMMY=0.6184 (paper≈0.6092).

- CANDIDATE_2, PhiRL vs B1_COMPOSITION_CHANGE: median Δ=-0.0005; ordering=False; Mann–Whitney p=0.7913; matrix-bootstrap 95% CI [-0.0030, -0.0003].
- CANDIDATE_2, PhiRL vs B2_RAW_COMPOSITIONS: median Δ=-0.0013; ordering=False; Mann–Whitney p=0.273; matrix-bootstrap 95% CI [-0.0060, -0.0012].
- CANDIDATE_2, PhiRL vs B3_MOLECULAR_FLUXES: median Δ=-0.0026; ordering=False; Mann–Whitney p=0.05381; matrix-bootstrap 95% CI [-0.0076, -0.0026].
- CANDIDATE_2, PhiRL vs D0_MAJORITY_DUMMY: median Δ=+0.3915; ordering=True; Mann–Whitney p=0.0001827; matrix-bootstrap 95% CI [+0.3742, +0.4136].
- CANDIDATE_3, PhiRL vs B1_COMPOSITION_CHANGE: median Δ=+0.0008; ordering=True; Mann–Whitney p=0.7913; matrix-bootstrap 95% CI [-0.0001, +0.0013].
- CANDIDATE_3, PhiRL vs B2_RAW_COMPOSITIONS: median Δ=-0.0007; ordering=False; Mann–Whitney p=0.8205; matrix-bootstrap 95% CI [-0.0023, +0.0004].
- CANDIDATE_3, PhiRL vs B3_MOLECULAR_FLUXES: median Δ=-0.0015; ordering=False; Mann–Whitney p=0.2411; matrix-bootstrap 95% CI [-0.0029, -0.0011].
- CANDIDATE_3, PhiRL vs D0_MAJORITY_DUMMY: median Δ=+0.3605; ordering=True; Mann–Whitney p=0.0001827; matrix-bootstrap 95% CI [+0.3424, +0.3840].

Exact paper-panel and broader directional gates were frozen separately. The primary machine classification above follows those gates in both candidates; favorable pooling was prohibited.

## Valid molecular-cell performance

- CANDIDATE_2: P1 valid-cell accuracy median=0.9732, balanced accuracy median=0.5027, AUPRC median=0.9834.
- CANDIDATE_3: P1 valid-cell accuracy median=0.9763, balanced accuracy median=0.5011, AUPRC median=0.9851.

The adjacent-H target was already positive by the quarter cutoff in nearly every matrix. Consequently, even accuracy on valid suffix cells is future-state occupancy rather than a scientifically eligible test of initial appearance. Completed-fit P1 also remains explicitly future-dependent; suffix perturbation changed P1 while P2 was invariant on every sentinel (`10/10`).

## Padding and length diagnostics

- CANDIDATE_2: all−valid=+0.0062; fraction correct from padding=0.3854; length/boundary within 0.03=True; shuffled-label retention=1.0000; padding-dominated=True.
- CANDIDATE_3: all−valid=+0.0041; fraction correct from padding=0.3552; length/boundary within 0.03=True; shuffled-label retention=1.0003; padding-dominated=True.

The registered majority, time, input-length, deterministic-boundary and transformed controls separate class prevalence, output position, boundary inference and molecular features. A padding-dominated result cannot support prediction of self-replication even if it resembles the paper's numerical boxplots.

## Figures

![01_trajectory_length_and_padding_extent](figures/01_trajectory_length_and_padding_extent.png)

*Figure 1. 01 trajectory length and padding extent.*

![02_padding_prevalence_geometry](figures/02_padding_prevalence_geometry.png)

*Figure 2. 02 padding prevalence geometry.*

![03_figure5_all_cell_reconstruction](figures/03_figure5_all_cell_reconstruction.png)

*Figure 3. 03 figure5 all cell reconstruction.*

![04_valid_cell_model_panel](figures/04_valid_cell_model_panel.png)

*Figure 4. 04 valid cell model panel.*

![05_four_mask_conditions](figures/05_four_mask_conditions.png)

*Figure 5. 05 four mask conditions.*

![06_accuracy_decomposition](figures/06_accuracy_decomposition.png)

*Figure 6. 06 accuracy decomposition.*

![07_length_time_and_dummy_diagnostics](figures/07_length_time_and_dummy_diagnostics.png)

*Figure 7. 07 length time and dummy diagnostics.*

![08_negative_controls](figures/08_negative_controls.png)

*Figure 8. 08 negative controls.*

![09_length_boundary_determinability](figures/09_length_boundary_determinability.png)

*Figure 9. 09 length boundary determinability.*

![10_future_dependence_and_suffix_invariance](figures/10_future_dependence_and_suffix_invariance.png)

*Figure 10. 10 future dependence and suffix invariance.*

![11_model_order_effects](figures/11_model_order_effects.png)

*Figure 11. 11 model order effects.*

![12_padding_dominance_matrix](figures/12_padding_dominance_matrix.png)

*Figure 12. 12 padding dominance matrix.*

![13_final_decision_matrix](figures/13_final_decision_matrix.png)

*Figure 13. 13 final decision matrix.*

## Validation, provenance and limitations

The method contract was committed and pushed before cohort generation. A new domain-separated seed root had zero detected overlap with prior seed material or input hashes. Incomplete/extinct/overflow units were retained under registered statuses and never replaced. All 400 trajectory identities, clocks, fission counts, matrix/initial-state hashes and complete feature arrays were independently regenerated from fresh caches. Model probability replay was exact for every registered sentinel fit. Temporary trajectories, tensors and model outputs remained under `/cache/e01_s19_l15`; only compact evidence was promoted. Technical amendment 001 repaired only Figure 11's input-table selection; amendment 002 repaired only a report Boolean-field lookup by rendering its already registered defining expression. Both failed partial assemblies remain quarantined, and all 24 frozen scientific hashes were exact before and after each amendment.

This remains adaptive forensic work on a preprint with missing padding, target and implementation semantics. Numerical resemblance cannot identify author code and cannot rescue failure on real cells, future independence, incremental value beyond H/stability, or causal control.

## Mandatory handoff

Stop here. No downstream step is active. The human reviewer may decide whether the result warrants a separately locked untouched confirmation, another narrow forensic question, author-code wait, S20 closeout/confirmation, E02 preparation, or pause.
