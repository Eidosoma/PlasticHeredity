# S19-L17 — BreakingGRNMemories Phi Lineage Transfer Audit

## Chief/human handoff

- **Step:** `E01-S19-L17-BREAKINGGRNMEMORIES-PHI-LINEAGE-TRANSFER-AUDIT-v1.0.0`
- **Status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Outcome classification:** `AUTHOR_AMBIGUITY_UNRESOLVED`, `NOT_PROMOTABLE`, `EXPLORATORY_NON_SUPPORT`, `POSSIBLE_STABILITY_PROXY`
- **Validation:** source-equivalence fixtures `4/4` passed; scientific replay `1`; immutable prior validation passed.
- **Artifacts:** source snapshot/history/license audit, executable dataflow, lineage/function/metric crosswalks, frozen hypothesis registry, fixture evidence, 200-trajectory conditional transfer evidence, controls, bootstrap summaries, replay/runtime/storage/hash manifests, and this report.
- **Lay summary:** No transferred scalar produced a directionally positive label association in both candidates.
- **Caveat:** BreakingGRNMemories is related-team source-lineage inspiration, not the unavailable GARD author implementation. Every scientific value is completed-fit and retrospective; the frozen label remains exactly determined by adjacent H.
- **Recommended next action:** mandatory human review. No L18, S20, E02, author contact, prediction, intervention, confirmation, or report generation is active.

## Authoritative human direction preserved verbatim

> We have plenty of time and we are in exploratory mode. Check this repo: https://github.com/pigozzif/BreakingGRNMemories
>
> This is the latest code we have from the team that did the paper we're trying to replicate and it has phi work as well which has been what we are not able to replicate - we can replicate the replicators, even too well - let this be an inspiration for L17

## Frozen question

Does the latest public BreakingGRNMemories Phi lineage specify a complete transferable pipeline, and—after source equivalence—does an unchanged transfer to the frozen S13Y cohort recover independent Phi evidence that was absent from prior E01 branches?

## Source audit anchor results

- Default branch `master` was frozen at commit `afe44231ad3ce915172cdb53a6b234bd76fcb6a5`, tree `56f66ab8b57a2c60e830370842926708eee0767d`.
- No license file, unit tests, Phi fixtures, or raw Phi input trajectories are present; public source is not redistributed.
- `information.py` is the corrected-IIGR lineage plus preprocessing/circuit wrappers. It differs materially from PhiRL through GSR and AR residualization, slow summed bidirectional MI, no active-variable filter, and unregularized covariance.
- Current `phi.py` exports only nonfinite-to-zero `emergence`; `information.compute_circuit_info` also exposes raw `emergence` and corrected `integrated/local_phi_r`.
- The tracked `info.txt` cannot be regenerated from any visible exact script state because its phase/measure schema conflicts with both the original and current scripts.
- No GARD adapter, self-replicator label, prefix refit, prediction model, or intervention scorer exists in this repository.

## Registered transfer hypotheses

Exactly three were frozen before GARD outcomes: current `phi.py` nonfinite-to-zero emergence, `optima.py`/information raw emergence, and information integrated/local-Phi-r. They share one source computation but retain distinct scalar and numerical identities. No prefix mode was run because the source does not specify one.

## Source equivalence

| fixtureId              | sourceStatus   | cleanStatus   | sourceReason   | cleanReason   | exactSourceReplayPassed   | statusEquivalencePassed   | fixturePassed   |
|:-----------------------|:---------------|:--------------|:---------------|:--------------|:--------------------------|:--------------------------|:----------------|
| COUPLED_GAUSSIAN       | ELIGIBLE       | ELIGIBLE      |                |               | True                      | True                      | True            |
| COUPLED_AUTOREGRESSIVE | ELIGIBLE       | ELIGIBLE      |                |               | True                      | True                      | True            |
| CONSTANT_CHANNEL       | ELIGIBLE       | ELIGIBLE      |                |               | True                      | True                      | True            |
| DUPLICATED_CHANNEL     | ELIGIBLE       | ELIGIBLE      |                |               | True                      | True                      | True            |

Synthetic fixtures were evaluated against the unmodified pinned public functions in an isolated process using the repository's exact NumPy/SciPy/NetworkX versions. The GARD transfer used the safe-JSON clean-room implementation already validated against the corrected IIGR lineage.

## Frozen cohort and execution

The exact S13Y 100-shared-matrix/200-trajectory candidate-2/candidate-3 cohort was used. No matrix, trajectory, label, threshold, exposure, feature tensor, model, or intervention was generated. Paper-frozen additive-0.5 closure, CLR99 and selected molecular clock were retained. Six workers used one numerical-library thread each under CPU float64.

Trajectory status counts: `{"ELIGIBLE":200}`.

## Primary run-level results

| candidateId       | hypothesisId                                   |   definedMatrices |   medianEstimate |   bootstrapLower95 |   bootstrapUpper95 |   positiveCount |   negativeCount |   medianExactHControlRho |
|:------------------|:-----------------------------------------------|------------------:|-----------------:|-------------------:|-------------------:|----------------:|----------------:|-------------------------:|
| S12F-CANDIDATE-02 | H1_BGM_CURRENT_PHI_EMERGENCE_NANZERO_COMPLETED |                99 |      -0.00110819 |        -0.0119596  |        0.0105086   |              48 |              51 |               0.00820127 |
| S12F-CANDIDATE-02 | H2_BGM_OPTIMA_EMERGENCE_RAW_COMPLETED          |                99 |      -0.00110819 |        -0.0149885  |        0.0116828   |              48 |              51 |               0.00820127 |
| S12F-CANDIDATE-02 | H3_BGM_INFORMATION_INTEGRATED_RAW_COMPLETED    |                99 |      -0.00810492 |        -0.0224846  |       -0.000368809 |              39 |              60 |              -0.0459148  |
| S12F-CANDIDATE-03 | H1_BGM_CURRENT_PHI_EMERGENCE_NANZERO_COMPLETED |                96 |       0.00495587 |        -0.00296713 |        0.0219808   |              56 |              39 |               0.0164461  |
| S12F-CANDIDATE-03 | H2_BGM_OPTIMA_EMERGENCE_RAW_COMPLETED          |                96 |       0.00495587 |        -0.00296713 |        0.0210154   |              56 |              39 |               0.0164461  |
| S12F-CANDIDATE-03 | H3_BGM_INFORMATION_INTEGRATED_RAW_COMPLETED    |                96 |      -0.00312738 |        -0.0139477  |        0.00563534  |              46 |              50 |              -0.0309038  |

Exact incoming H, continuous composition change, and the frozen adjacent-H label were retained side by side. A label association is not independent replication evidence merely because it resembles the paper: `Y=I(H>0.9)` exactly, and completed-fit BGM values use the full future suffix.

## Metric and temporal interpretation

L12's metric adjudication remains unchanged: the paper equation, public `emergence`, and public `integrated/local_phi_r` are not algebraically interchangeable. BreakingGRNMemories strengthens the public lineage for both names but does not resolve which GARD scalar the authors used. Its public pipeline is global/completed-fit; L17 therefore supplies no future-suffix-independent or prospective evidence.

## Numerical and provenance findings

- CPU float64 was authoritative; GPU time was zero.
- The source's unregularized Gaussian entropy was retained. Source exceptions and partial nonfinite arrays were status-bearing and never repaired.
- The public caller does not fully seed Python, NumPy, or the NetworkX Fiedler initialization. L17's domain-separated NumPy seed is an explicit exact-replay wrapper, not evidence of author seed identity.
- Every unit was recomputed into a separate replay cache and compared at array-hash, partition, status, and reason level.
- Technical amendment 001 corrected only an unused frozen-clock metadata attribute and empty-summary serialization after preserving the failed attempt; technical amendment 002 added the registered header to a zero-row failure ledger. Neither amendment changed a source path, trajectory, seed, scalar, metric, gate, classification, or scientific value.
- Runtime: `904.731` wall seconds, `0.948480` reported worker CPU-hours; retained artifacts and temporary cache remained below their hard ceilings.

## Limitations

This is a post hoc exploratory transfer from a related paper and team lineage. The repository is newer than the target preprint, does not contain the GARD analysis, lacks a license file and reproducible Phi fixture, and has a tracked-output provenance inconsistency. The paper-to-BGM CLR adapter and replay seed wrapper are explicit reconstruction choices. Neither a favorable nor unfavorable transfer can identify the unavailable author code.

## Mandatory human-review boundary

Stop here. All L17 outputs are frozen. No L18, S20, E02, author contact, new simulation, prediction, intervention, confirmation, or report-bundle generation begins automatically.
