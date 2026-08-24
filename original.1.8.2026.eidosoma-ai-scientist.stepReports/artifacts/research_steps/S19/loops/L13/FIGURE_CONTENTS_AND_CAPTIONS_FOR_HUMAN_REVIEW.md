# L13 Output-Figure Contents and Captions for Human Review — V1

## Top summary

- **Research step ID:** `S19-L13` (`E01-S19-L13-FIGURE5-RECURRING-TARGET-PREDICTION-RECONSTRUCTION-v1.0.0`)
- **Version boundary:** `V1_GENERATED_L13_FIGURES`. This file documents L13's generated result plots; it is retained for provenance but is not the requested reading of the input paper. Use `FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md` for that purpose.
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Artifacts covered:** 14 registered PNG figures under `figures/`, each embedded below with a SHA-256 identity.
- **Validation result:** All listed image files exist; captions describe only frozen L13 tables and gates; image hashes are recorded for eye-checking and artifact identity.
- **Outcome classification:** `FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED`; promotion `NOT_PROMOTABLE`.
- **Caveats or blockers:** These are forensic reconstruction plots, not native paper panels. R1/U2 targets are completed-run definitions, P1 is future-dependent, Figure 5 ranges are approximate L12 digitizations, and no image identifies author code.
- **Recommended next action:** Inspect the figures below, compare the visible relationships with their captions, and return a human decision. Keep S20, E02, confirmation, and L14 inactive.

## How to review

The **Contents** line states what was plotted. The **Caption** states the evidentiary interpretation. The **Visual check** calls out what should be apparent by eye. If an image appears inconsistent with its visual check, use its SHA-256 below to identify the exact file for review; the machine-readable tables remain authoritative.

## Figure 1. Baseline arithmetic clue

![Figure 1: Baseline arithmetic clue](figures/figure_01_baseline_arithmetic_clue.png)

- **File:** `figures/figure_01_baseline_arithmetic_clue.png`
- **SHA-256:** `b504ff02732787aa24bbc504950f488cefdbaef7bfa5dc1caaca67e9886c3b27`
- **Contents:** The left panel plots mean whole-run occupancy against mean per-matrix majority accuracy for each label and candidate; the right panel keeps the ten split-wise S16 dummy distributions separate for R1/U2 and candidate 2/3.
- **Caption:** The left panel plots mean whole-run occupancy against mean per-matrix majority accuracy for each label and candidate; the right panel keeps the ten split-wise S16 dummy distributions separate for R1/U2 and candidate 2/3. The left values near 0.75 are not a contradiction: averaging each matrix's majority accuracy is nonlinear and exposes strong between-matrix prevalence heterogeneity. The right panel is the registered task-level gate. Its green band is the frozen 0.55–0.65 Figure 5 envelope; verify that both U2 candidate medians, but neither R1 candidate median, lie inside it.
- **Visual check:** The left values near 0.75 are not a contradiction: averaging each matrix's majority accuracy is nonlinear and exposes strong between-matrix prevalence heterogeneity. The right panel is the registered task-level gate. Its green band is the frozen 0.55–0.65 Figure 5 envelope; verify that both U2 candidate medians, but neither R1 candidate median, lie inside it.

## Figure 2. Target availability

![Figure 2: Target availability](figures/figure_02_target_availability.png)

- **File:** `figures/figure_02_target_availability.png`
- **SHA-256:** `9cb3dbbe9f9670c0319a6c07ddc2d1579a35860560162a2c4bc229a9157034fa`
- **Contents:** Defined recurring-target sequences are counted out of 100 matrices for each target and candidate.
- **Caption:** Defined recurring-target sequences are counted out of 100 matrices for each target and candidate. The dashed 80-matrix line is the forensic-evaluation threshold and the dotted 95-matrix line is the paper-scale promotion threshold. Verify R1 at 89/86 and U2 at 100/100.
- **Visual check:** The dashed 80-matrix line is the forensic-evaluation threshold and the dotted 95-matrix line is the paper-scale promotion threshold. Verify R1 at 89/86 and U2 at 100/100.

## Figure 3. Whole-run versus suffix prevalence

![Figure 3: Whole-run versus suffix prevalence](figures/figure_03_whole_suffix_prevalence.png)

- **File:** `figures/figure_03_whole_suffix_prevalence.png`
- **SHA-256:** `0c803e0d99d8812447e1589aa6280d4dd9674c80721c8b7285bd9f94998dac68`
- **Contents:** Mean positive prevalence is shown for complete molecular trajectories and the final 75% prediction suffix.
- **Caption:** Mean positive prevalence is shown for complete molecular trajectories and the final 75% prediction suffix. Verify that target prevalence is not manufactured by padding and that the U2 suffix remains in the broad range capable of producing a roughly 60% dummy.
- **Visual check:** Verify that target prevalence is not manufactured by padding and that the U2 suffix remains in the broad range capable of producing a roughly 60% dummy.

## Figure 4. First-onset eligibility at the 25% cutoff

![Figure 4: First-onset eligibility at the 25% cutoff](figures/figure_04_first_onset_availability.png)

- **File:** `figures/figure_04_first_onset_availability.png`
- **SHA-256:** `12259e7f6e27d78e94d01bd86c2d1e84d4b3cc6d8ed5f2d365ac3dee4e2cd3bd`
- **Contents:** Bars show the fraction still pre-onset at the cutoff and the fraction whose first positive target state occurs in the suffix.
- **Caption:** Bars show the fraction still pre-onset at the cutoff and the fraction whose first positive target state occurs in the suffix. The visually small bars are decisive: only 3 candidate-2 and 5 candidate-3 U2 matrices meet both onset concepts, far below the required 20 per candidate.
- **Visual check:** The visually small bars are decisive: only 3 candidate-2 and 5 candidate-3 U2 matrices meet both onset concepts, far below the required 20 per candidate.

## Figure 5. Representative recurring-target sequences

![Figure 5: Representative recurring-target sequences](figures/figure_05_representative_target_sequences.png)

- **File:** `figures/figure_05_representative_target_sequences.png`
- **SHA-256:** `46ea12468ac9ee059b84502bd82e05e8eb90172af362c8d5d18a47760311d552`
- **Contents:** Direct molecular R1 and U2 labels are shown for one eligible trajectory per candidate; adjacent-H occupancy distributions provide the frozen high-prevalence contrast.
- **Caption:** Direct molecular R1 and U2 labels are shown for one eligible trajectory per candidate; adjacent-H occupancy distributions provide the frozen high-prevalence contrast. Look for episode timing and switching rather than occupancy alone. The recurring labels are not boundary projections, and the adjacent-H comparator should appear much more uniformly positive across matrices.
- **Visual check:** Look for episode timing and switching rather than occupancy alone. The recurring labels are not boundary projections, and the adjacent-H comparator should appear much more uniformly positive across matrices.

## Figure 6. Completed-fit versus prefix-only PhiRL inputs

![Figure 6: Completed-fit versus prefix-only PhiRL inputs](figures/figure_06_completed_vs_prefix_phirl.png)

- **File:** `figures/figure_06_completed_vs_prefix_phirl.png`
- **SHA-256:** `6bdb0a26d9f98090d8b75aaec14bd4dc1d52787bfa8c2f111f1094245cade737`
- **Contents:** First-quarter emergence values for P1 (fit on the completed trajectory) and P2 (fit only on the first quarter) are overlaid for one matrix in each candidate.
- **Caption:** First-quarter emergence values for P1 (fit on the completed trajectory) and P2 (fit only on the first quarter) are overlaid for one matrix in each candidate. Verify that these are materially different feature trajectories even though they use the same observed prefix; this is the visible signature of completed-fit future dependence.
- **Visual check:** Verify that these are materially different feature trajectories even though they use the same observed prefix; this is the visible signature of completed-fit future dependence.

## Figure 7. Reconstructed Figure 5 accuracy boxplots

![Figure 7: Reconstructed Figure 5 accuracy boxplots](figures/figure_07_paper_accuracy_boxplots.png)

- **File:** `figures/figure_07_paper_accuracy_boxplots.png`
- **SHA-256:** `2bb027b1547f3bbf3bbad2d5ff4c5ef66073dfb72bce34f2ba4ed4c98597033a`
- **Contents:** Ten split-wise accuracies compare completed-fit PhiRL, composition change, raw composition, flux, and the majority dummy for the advancing U2 target in each candidate.
- **Caption:** Ten split-wise accuracies compare completed-fit PhiRL, composition change, raw composition, flux, and the majority dummy for the advancing U2 target in each candidate. The dotted 0.85 and dashed 0.60 guides mark the digitized PhiRL and dummy centers. Verify that the dummy is reconciled but completed-fit PhiRL remains well below the paper-like 0.85 range and does not yield the full registered ordering/significance pattern.
- **Visual check:** The dotted 0.85 and dashed 0.60 guides mark the digitized PhiRL and dummy centers. Verify that the dummy is reconciled but completed-fit PhiRL remains well below the paper-like 0.85 range and does not yield the full registered ordering/significance pattern.

## Figure 8. Robust discrimination and calibration-loss metrics

![Figure 8: Robust discrimination and calibration-loss metrics](figures/figure_08_robust_metrics.png)

- **File:** `figures/figure_08_robust_metrics.png`
- **SHA-256:** `20e0eaf3c71540c86ee4d415bc112e9ba8f9c3a308427cbee4ef8b88ef1db5ac`
- **Contents:** Median balanced accuracy, AUROC, AUPRC, and Brier score are displayed for completed-fit PhiRL, prefix-only PhiRL, and the dummy by candidate.
- **Caption:** Median balanced accuracy, AUROC, AUPRC, and Brier score are displayed for completed-fit PhiRL, prefix-only PhiRL, and the dummy by candidate. Accuracy alone can ride prevalence. Verify that prefix-only balanced accuracy stays near chance and interpret lower Brier as better, unlike the three higher-is-better discrimination metrics.
- **Visual check:** Accuracy alone can ride prevalence. Verify that prefix-only balanced accuracy stays near chance and interpret lower Brier as better, unlike the three higher-is-better discrimination metrics.

## Figure 9. Incremental-value comparisons

![Figure 9: Incremental-value comparisons](figures/figure_09_incremental_value.png)

- **File:** `figures/figure_09_incremental_value.png`
- **SHA-256:** `f3a7142e57a21a8e6a126e7177111162f2a3f9076032bc1eab1b3f7637bad6cb`
- **Contents:** Paired catalytic-matrix bootstrap differences compare prefix-only PhiRL with ordinary controls and its registered combined models.
- **Caption:** Paired catalytic-matrix bootstrap differences compare prefix-only PhiRL with ordinary controls and its registered combined models. Intervals crossing zero fail incremental value. Verify that P2 does not consistently beat adjacent H or prefix-only attractor geometry in either candidate.
- **Visual check:** Intervals crossing zero fail incremental value. Verify that P2 does not consistently beat adjacent H or prefix-only attractor geometry in either candidate.

## Figure 10. Completed-fit dependence and prefix suffix-invariance

![Figure 10: Completed-fit dependence and prefix suffix-invariance](figures/figure_10_future_dependence.png)

- **File:** `figures/figure_10_future_dependence.png`
- **SHA-256:** `ef44535959ffc11c595cc6e9dfa364e85797fff3f6bc56a4dcbcfaa39350e99a`
- **Contents:** Sentinel suffix perturbations show nonzero changes in completed-fit P1 while exact-replay P2 markers remain at zero.
- **Caption:** Sentinel suffix perturbations show nonzero changes in completed-fit P1 while exact-replay P2 markers remain at zero. Verify both sides of the leakage audit: P1 changes substantially when the unseen suffix changes, while P2 remains exactly invariant.
- **Visual check:** Verify both sides of the leakage audit: P1 changes substantially when the unseen suffix changes, while P2 remains exactly invariant.

## Figure 11. Registered negative controls

![Figure 11: Registered negative controls](figures/figure_11_negative_controls.png)

- **File:** `figures/figure_11_negative_controls.png`
- **SHA-256:** `a72fb2ff3860519d8e3a7781648e21db0361fdc61c33df92a2783c4980b7e94c`
- **Contents:** Paired P2 accuracy advantages over temporal permutation and matrix-label permutation controls are shown with matrix-bootstrap intervals.
- **Caption:** Paired P2 accuracy advantages over temporal permutation and matrix-label permutation controls are shown with matrix-bootstrap intervals. A credible prospective signal should separate from every control. Verify that the registered control contrasts do not jointly establish such separation in both candidates.
- **Visual check:** A credible prospective signal should separate from every control. Verify that the registered control contrasts do not jointly establish such separation in both candidates.

## Figure 12. Candidate-2 versus candidate-3 agreement

![Figure 12: Candidate-2 versus candidate-3 agreement](figures/figure_12_candidate_agreement.png)

- **File:** `figures/figure_12_candidate_agreement.png`
- **SHA-256:** `bff53043527c7aada26d2d70d7a4d9b94368702e58121587ed6d7490dd528acc`
- **Contents:** Each point compares a model's median accuracy under candidate 2 and candidate 3; the diagonal denotes equality.
- **Caption:** Each point compares a model's median accuracy under candidate 2 and candidate 3; the diagonal denotes equality. Look for directional consistency without treating proximity to the diagonal as paper replication. The cross-candidate check cannot rescue failed paper-facing or prospective gates.
- **Visual check:** Look for directional consistency without treating proximity to the diagonal as paper replication. The cross-candidate check cannot rescue failed paper-facing or prospective gates.

## Figure 13. Retrospective and prospective gate matrix

![Figure 13: Retrospective and prospective gate matrix](figures/figure_13_decision_matrix.png)

- **File:** `figures/figure_13_decision_matrix.png`
- **SHA-256:** `0352cf2c13554e2416f2b46469bef85ada123cdf73ce4ce48860aff675ecf3bd`
- **Contents:** Green and red cells summarize every registered gate by target and candidate.
- **Caption:** Green and red cells summarize every registered gate by target and candidate. Verify that dummy overlap and suffix invariance pass, while the paper-interval, baseline-compatibility, significance, incremental-value, balanced-accuracy, and onset-eligibility requirements prevent promotion.
- **Visual check:** Verify that dummy overlap and suffix invariance pass, while the paper-interval, baseline-compatibility, significance, incremental-value, balanced-accuracy, and onset-eligibility requirements prevent promotion.

## Figure 14. Final L13 decision tree

![Figure 14: Final L13 decision tree](figures/figure_14_promotion_decision_tree.png)

- **File:** `figures/figure_14_promotion_decision_tree.png`
- **SHA-256:** `47b6c169a633c62059dfd5a7cf7c278e737ab515a9878525e3b5209d8a1f9a89`
- **Contents:** The locked flow from two recurring targets through geometry, retrospective completed-fit, and prospective prefix-only gates ends at the machine-authoritative classification.
- **Caption:** The locked flow from two recurring targets through geometry, retrospective completed-fit, and prospective prefix-only gates ends at the machine-authoritative classification. Verify the terminal result `FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED`, promotion `NOT_PROMOTABLE`, and mandatory human-review boundary.
- **Visual check:** Verify the terminal result `FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED`, promotion `NOT_PROMOTABLE`, and mandatory human-review boundary.
