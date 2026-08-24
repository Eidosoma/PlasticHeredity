# S19-L16 Full Results — Figure-5 Tensor and Architecture Discrimination

## Top summary

- **Research step:** `E01-S19-L16-FIGURE5-TENSOR-ARCHITECTURE-DISCRIMINATION-v1.0.0`
- **Completion status:** `COMPLETE_SOURCE_GROUNDING_GATE_STOP_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Outcome classification:** `AUTHOR_AMBIGUITY_UNRESOLVED`; `EXPLORATORY_NON_SUPPORT`; `NOT_PROMOTABLE`.
- **Directed decision:** `NO_SUFFICIENTLY_SOURCE_GROUNDED_COMPLETE_TENSOR_OR_ARCHITECTURE_HYPOTHESIS`.
- **Model execution:** No new model was fitted. The prospectively locked gate required this stop because zero audited conventions completely specified the tensor, variable-length handling, target representation, loss/scoring masks, aggregation, topology, and capacity from paper or Figure-5-linked public code.
- **Validation:** 2270 immutable prior files passed; 12/12 frozen feature tensors (36 arrays), 400/400 target/mask pairs, 10/10 splits, 60/60 frozen L15 artifact hashes, and 24/24 cached model/replay probability identities passed. Read-only all-cell, valid-cell, decomposition, length/time/boundary, permutation, and suffix-invariance evidence was copied byte-for-byte. One preserved validator-only amendment corrected the treatment of L15's two intentionally null ineligible-row hashes and changed no array, status, gate, or scientific value.
- **Recommended next action:** Mandatory human review. Public evidence remains insufficient for another scientifically locked Figure-5 model run; an exact author-code/configuration release or a separately authorized closeout decision would add more information than another adaptive tensor guess.

## Frozen question and pre-outcome decision rule

L16 asked whether the manuscript or pinned public lineage contains enough explicit detail to register at most three complete tensor/architecture hypotheses on the frozen L15 cohort. The lock required direct paper specification or public code explicitly linked to the GARD Figure-5 task for ten identity-changing fields. Plotting-only interpolation, generic reinforcement-learning feature extractors, and E01's own S16 implementation were deliberately insufficient substitutes for author grounding.

The complete contract was committed and pushed at `d737edc71b8ebd19b51bf5f2fd36b91e8133dc15` before this gate was evaluated.

## What the paper and public code actually specify

The manuscript directly specifies: the first 25% of the Phi-r trajectory as input; the remaining 75% self-replication trajectory as target; an MLP; an 80/20 split across runs; ten seeded repetitions; binary accuracy; and a majority-label dummy. It also states that molecular trajectory lengths vary stochastically.

It does **not** specify: the fixed tensor shape; interpolation/resampling/truncation; input or target padding; padding values; target mask; training loss weighting; scoring mask; per-cell versus per-run aggregation; MLP layer topology; capacity; activation; optimizer; or early stopping.

Pinned PhiRL contains a plotting helper that linearly interpolates reward curves to 1,000 points over the shortest seed horizon, plus generic RL vector/sequence feature extractors. The complete public Git history contains no GARD/self-replication 25%-to-75% prediction implementation. The interpolation is not a prediction tensor, and the extractors do not implement a GARD target decoder, mask, loss, accuracy, or complete supervised architecture.

## Audited candidate conventions

- `H0_FROZEN_S16_RIGHT_PAD_FLAT_MLP`: 0/10 required fields directly grounded; execution registration = `False`.
- `H1_COMMON_MINIMUM_HORIZON_INTERPOLATED_MLP`: 0/10 required fields directly grounded; execution registration = `False`.
- `H2_PUBLIC_PHIRL_GENERIC_VECTOR_OR_SEQUENCE_EXTRACTOR`: 0/10 required fields directly grounded; execution registration = `False`.

No convention was eligible. Combining the plotting interpolation with the generic extractor and S16's invented target/loss conventions would mix independently convenient components without a paper or source link—the exact architecture tournament the human contract prohibited.

## Frozen L15 evidence accepted without reinterpretation

- CANDIDATE_2: frozen L15 S11 medians were PhiRL `0.9786`, composition change `0.9786`, raw composition `0.9796`, flux `0.9810`, and dummy `0.5836`.
- CANDIDATE_3: frozen L15 S11 medians were PhiRL `0.9784`, composition change `0.9782`, raw composition `0.9793`, flux `0.9797`, and dummy `0.6184`.

Those values remain L15 evidence. L16 did not refit, tune, select, or reopen them. The frozen valid-cell balanced accuracy remains approximately 0.50, the length/boundary diagnostics remain near-deterministic, valid-label permutation retains the all-cell result, P1 remains completed-fit future-dependent, and P2 retains suffix invariance. These facts motivate the ambiguity but cannot identify a missing author tensor.

## Why no model run is the scientifically valid L16 result

The paper-panel accuracies lie between L15's valid-cell non-discrimination and unmasked length/padding near-determinism. Many unreported transformations could interpolate between those extremes. Without source grounding, executing normalized-time resampling, common-horizon truncation, run-level aggregation, or another MLP layout would select among underdetermined implementations using the already known panel. That would increase specification multiplicity rather than discriminate a paper-directed hypothesis.

This is a protocol gate result, not an operational failure. `LOOP_FAILED_CLOSED` is therefore not used. The correct classifications are `AUTHOR_AMBIGUITY_UNRESOLVED`, `EXPLORATORY_NON_SUPPORT`, and `NOT_PROMOTABLE`.

## Validation and regeneration

- Immutable prior baseline: `2270` files, `0` mismatches.
- Frozen artifact replay: `60/60` passed.
- Feature tensor replay: `36/36` arrays passed.
- Target replay: `400/400` matrix/candidate target-mask pairs passed.
- Split replay: `10/10` exact matrix-grouped 128/32/40 splits passed.
- Cached model identity: `24/24` original/replay probability hashes passed; no L16 training was run.
- Read-only L15 machine tables were copied byte-identically and their accuracy decomposition remains explicit.
- Technical amendment 001 preserves the first failed partial attempt under `/cache/e01_s19_l16/failed_attempt_001_artifacts`; it changes only the independent validator's handling of the two registered ineligible rows, whose manifest hashes are intentionally null.
- CPU float64 remained authoritative; GPU use was zero; no matrix, trajectory, label, feature, split, model outcome, intervention, or report bundle was generated.

## Figures

![Direct source-grounding matrix](figures/01_source_grounding_matrix.png)

*Figure 1. Direct support across the ten identity-changing fields. Partial paper descriptions, plotting-only code, generic RL extractors, and frozen E01 choices do not satisfy the direct Figure-5 grounding gate.*

![Hypothesis completeness](figures/02_hypothesis_completeness.png)

*Figure 2. None of the three audited convention candidates becomes an executable complete hypothesis.*

![Frozen L15 panel gap](figures/03_frozen_l15_panel_gap.png)

*Figure 3. Read-only L15 all-cell results bracket a task-geometry problem but do not identify a source-supported intermediate convention.*

![Gate-stop path](figures/04_gate_stop_decision_path.png)

*Figure 4. The prospectively locked source-grounding gate stops before new model fitting.*

## Interpretation boundary

L16 does not show that no private coherent implementation exists. It shows that the inspected manuscript and public lineages cannot completely distinguish one. No result identifies author code, supports prospective initial-appearance prediction, changes the completed-fit leakage finding, supports intervention or causal control, or changes any S18 classification.

## Mandatory human-review boundary

Stop here. L16 is frozen. S20, E02, author contact, confirmation, intervention work, report generation, and any later loop remain inactive pending a new explicit human decision.
