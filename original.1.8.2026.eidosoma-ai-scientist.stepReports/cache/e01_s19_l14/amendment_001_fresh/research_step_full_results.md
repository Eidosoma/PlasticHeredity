# S19 Current-Step Handoff — S19-L14

## Top summary

- **Research step:** `E01-S19-L14-FIGURE5-PADDING-LENGTH-LEAKAGE-RECONSTRUCTION-v1.0.0`
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Outcome classification:** `FIGURE5_PADDING_ARITHMETIC_NOT_SUPPORTED`; `FIGURE5_PADDING_ARITHMETIC_NOT_SUPPORTED, NOT_PROMOTABLE`
- **Artifacts written:** complete required L14 machine-readable evidence, 15 figures, validation and hash manifests, this report, and the S19 current-step handoff.
- **Validation result:** immutable prior, paper/source identities, all 16 fixtures, exact S16 target/feature replay, tensor scope, serialization, storage, and regeneration passed.
- **Central caveat:** this is adaptive forensic reconstruction. All-cell padding accuracy is not molecular-state prediction and cannot alter S18 prospective or causal-control conclusions.
- **Recommended next action:** mandatory human review. Keep S20, E02, author contact, confirmation matrices, interventions, and another S19 loop inactive.

## Frozen question

Could the exact S16 adjacent-incoming `H>0.9` task, with right-padded target zeros included in loss and/or accuracy, explain Figure 5's approximately 0.61 dummy and approximately 0.79–0.85 model accuracies? The simulator, trajectories, labels, features, model, splits, quarter cutoff, and padding values were held fixed.

## Inputs and provenance

The analysis used exactly 100 paired S13Y/S16 matrices per candidate, the frozen selected molecular clocks, 100 completed fissions, the S16 64/16/20 matrix splits across ten repetitions, and CPU-float64 PhiRL features. The original paper hash was `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`. No new matrix, trajectory, label, feature, model, metric, or intervention result was generated.

## Human-panel and digitization lock

The frozen panel audit treats Figure 5 as five boxplots and its isolated raw-composition circle as a flier. Pixel calibration was fixed from y-axis ticks before cohort arithmetic. Figure 2 endpoints were recorded as approximate molecular-step constraints; terminal aggregate support remains potentially sparse. The locked details are in `human_panel_review_lock.yaml`, `paper_figure5_digitization_lock.csv`, and `paper_figure2_length_lock.csv`.

## Mandatory fixtures and exact S16 replay

All 16 fixtures passed. The full tensor regeneration replayed 1,200 candidate/matrix/feature identities against S16 hashes (six features × 200 trajectories), including target clocks, right padding, feature masks, and first-quarter-only PhiRL. No mismatch was accepted.

## Padding arithmetic

The required identity `p_padded = p_valid × q_valid` held to machine precision.

- S12F-CANDIDATE-02: q=0.5970, valid prevalence=0.9823, padded prevalence=0.5864, padded dummy=0.5864.
- S12F-CANDIDATE-03: q=0.6342, valid prevalence=0.9843, padded prevalence=0.6242, padded dummy=0.6242.

The human-directed reference calculation `0.61/0.88 = 0.6932` is reported as an interpretive arithmetic clue, not exact author data. With the frozen adjacent-H target, valid prevalence is measured directly rather than substituted with Table 1's 0.88.

## Advancement adjudication

- S12F-CANDIDATE-02: split dummy median=0.5969; paper IQR=0.6008–0.6208; q compatible=False; gate=False.
- S12F-CANDIDATE-03: split dummy median=0.6390; paper IQR=0.6008–0.6208; q compatible=False; gate=False.

The prospectively locked arithmetic gate failed, so the protocol prohibited MLP execution. Empty, schema-valid downstream tables and explicit not-executed figures preserve complete scope accounting without opening another convention.

## Four mask conditions

- `S00`: masked training, masked scoring (exact S16 scientific condition).
- `S01`: masked training, unmasked all-cell scoring (score-inflation isolation).
- `S10`: unmasked padded training, masked valid-cell scoring (training-contamination isolation).
- `S11`: unmasked padded training and scoring (primary forensic convention).

## Interpretation boundaries

- A numerical all-cell panel match would only show that a padding convention can reconstruct Figure 5.
- Padding zeros are not self-replicating or non-self-replicating molecular observations.
- Completed-fit PhiRL remains future-dependent.
- The adjacent-H label is effectively determined by exact H and provides almost no genuine pre-onset cohort at the cutoff.
- L12 remains `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION`; L13 remains `FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED`; S18 prospective prediction and causal-control non-support remain unchanged.

## Figures

![figure01_trajectory_lengths_and_padding_extent](loops/L14/figures/figure01_trajectory_lengths_and_padding_extent.png)

*1. figure01 trajectory lengths and padding extent.*
![figure02_required_valid_fraction](loops/L14/figures/figure02_required_valid_fraction.png)

*2. figure02 required valid fraction.*
![figure03_real_vs_padded_prevalence](loops/L14/figures/figure03_real_vs_padded_prevalence.png)

*3. figure03 real vs padded prevalence.*
![figure04_dummy_valid_vs_padded](loops/L14/figures/figure04_dummy_valid_vs_padded.png)

*4. figure04 dummy valid vs padded.*
![figure05_four_mask_conditions](loops/L14/figures/figure05_four_mask_conditions.png)

*5. figure05 four mask conditions.*
![figure06_reconstructed_figure5_all_cells](loops/L14/figures/figure06_reconstructed_figure5_all_cells.png)

*6. figure06 reconstructed figure5 all cells.*
![figure07_reconstructed_figure5_valid_cells](loops/L14/figures/figure07_reconstructed_figure5_valid_cells.png)

*7. figure07 reconstructed figure5 valid cells.*
![figure08_accuracy_decomposition](loops/L14/figures/figure08_accuracy_decomposition.png)

*8. figure08 accuracy decomposition.*
![figure09_length_and_boundary_controls](loops/L14/figures/figure09_length_and_boundary_controls.png)

*9. figure09 length and boundary controls.*
![figure10_valid_label_shuffle](loops/L14/figures/figure10_valid_label_shuffle.png)

*10. figure10 valid label shuffle.*
![figure11_input_length_obfuscation](loops/L14/figures/figure11_input_length_obfuscation.png)

*11. figure11 input length obfuscation.*
![figure12_completed_vs_prefix](loops/L14/figures/figure12_completed_vs_prefix.png)

*12. figure12 completed vs prefix.*
![figure13_candidate_agreement](loops/L14/figures/figure13_candidate_agreement.png)

*13. figure13 candidate agreement.*
![figure14_decision_matrix](loops/L14/figures/figure14_decision_matrix.png)

*14. figure14 decision matrix.*
![figure15_promotion_decision_tree](loops/L14/figures/figure15_promotion_decision_tree.png)

*15. figure15 promotion decision tree.*

## Validation and reproducibility

Repository code/config/tests were locked and pushed before cohort arithmetic. Temporary tensors and model intermediates remained under `/cache/e01_s19_l14`; compact evidence only was promoted. `immutable_prior_validation.json`, `regeneration_validation.json`, `storage_validation.json`, and `artifact_manifest.json` provide the final checks. Technical amendment 001 corrected only the root-handoff figure-link prefix after final validation exposed the report-assembly defect; all scientific tables, gates, values and classifications remained hash-identical.

## Mandatory handoff

Stop here for human review. No L15, S20, E02, author contact, untouched confirmation, intervention, or report bundle has been activated.
