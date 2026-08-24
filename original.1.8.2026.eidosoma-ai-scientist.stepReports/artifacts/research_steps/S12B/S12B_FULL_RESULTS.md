# E01-S12B Source-Code Reconstruction and Future-Dependence Audit of Local Phi-r

## Top summary

- **Research step ID:** S12B (`E01-S12B-PIGOZZI-SOURCE-CODE-AUDIT-v1.0.0`)
- **Completion status:** STOPPED_AT_PREREGISTERED_GATE
- **Artifacts written:** `S12B_FULL_RESULTS.md`, `artifact_manifest.json`, `classification.json`, `failure_ledger.csv`, `figures/association_distributions.png`, `figures/final_decision_matrix.png`, `figures/full_trajectory_matched_sources.png`, `figures/full_versus_prefix_representative.png`, `figures/partition_stability.png`, `figures/spike_overlap.png`, `full_trajectory_local_values.parquet`, `future_dependence_results.csv`, `immutable_input_audit.json`, `partition_history.parquet`, `prefix_endpoint_values.parquet`, `preregistration.yaml`, `preregistration_record.json`, `prospective_associations.csv`, `research_step_full_results.md`, `retrospective_associations.csv`, `runtime_manifest.json`, `safe_phi_lattice.json`, `source_audit.md`, `source_diagnostic_outputs.parquet`, `source_equivalence_results.csv`, `source_snapshot_manifest.json`, `spike_analysis.csv`, `status.json`
- **Validation result:** FAIL CLOSED — SOURCE_EQUIVALENCE_FAILED
- **Outcome classification:** constraining/contradictory; decision `SOURCE_RECONSTRUCTION_FAILED`.
- **Caveats or blockers:** `IIGR_CORRECTED_SOURCE` disagreed with the untouched source adapter on the deliberately singular fixture: the source returned `ELIGIBLE`, while the clean-room wrapper returned `INELIGIBLE_SOURCE_PIPELINE_EXCEPTION`.
- **Lay summary:** The safety check failed before any GARD trajectory was opened. Ordinary synthetic arrays matched to far tighter than the frozen numerical tolerances, but the IIGR wrapper did not reproduce the source's behavior on the required singular edge case. The audit therefore produced no full-trajectory or prefix Phi-r result and attempted no repair.
- **Recommended next action:** Return for human review. Do not repair after outcomes, do not begin S13, and preserve the failed source audit.

## Frozen question

Do the exact public `IntegratedInformationGeneRegulation` and `PhiRL` source behaviors recover paper-like punctuated local Phi-r and positive replication association on the twelve existing GARD runs, and do those patterns survive when the same implementation is refitted using past-only prefixes?

## Inputs

The twelve immutable S12 baseline trajectories, historical S08/S12 labels, matrices/provenance, additive-0.5 dropped-component 99-dimensional CLR substrate, and original paper were registered as frozen inputs. Their hashes and the complete S10, S11, S11R, and S12 directory identities were verified before and after the equivalence stage. Because equivalence failed first, the runner did **not** open or preprocess any S12 trajectory table. No new GARD or intervention trajectory was generated.

Pinned public sources were `pigozzif/IntegratedInformationGeneRegulation@7c1c22fe39f539d4a453135476f1f0dd5a6b45f7` and `pigozzif/PhiRL@a6d1d0d18c7551302724b7158c6ccdc4d3a33373`; PhiRL regularization commit `9030b598f436cd23c39a3c3fc312ff79c79fb2ad` was verified as an ancestor. No license file was detected in either pinned tree. The public code is not the unavailable GARD implementation.

## Detailed methods

Counts were closed with delta 0.5, CLR transformed, and original component 100 removed. All materialized initial, post-molecular-event, and selected post-fission observations remained in order. IIGR used source z-scoring, global-signal regression, lag-one residualization, a second z-score, bidirectional lag-one Gaussian MI, the source unnormalized Fiedler sign split with a `1e-6` graph floor, partition averages, and corrected all-atom `local_phi_r`. PhiRL removed dimensions with standard deviation at or below `1e-8`, z-scored, used its fast MI/Fiedler pipeline, and used trace-scaled covariance regularization with epsilon `1e-6`. CPU float64 was authoritative.

Full mode fitted preprocessing, partition, means, and covariances once to the completed trajectory and labeled every value `RETROSPECTIVE_FULL_TRAJECTORY_LOCAL`. Prefix mode refitted the identical source behavior independently at each post-fission point with at least 256 preceding molecular transitions and retained only the endpoint value. Current-generation Spearman association was the preregistered primary estimand; next-generation association was secondary. Trajectory bootstrap and within-trajectory circular-shift nulls each used 4,096 replicates. Completed-fit and prefix values were also compared at shared fissions and over a direct first-quarter refit.

## Commands

```text
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -I scripts/e01/convert_s12b_phi_lattice.py ...
PYTHONPATH=src pytest -q tests/e01/test_pigozzi_source_audit.py tests/e01/test_s12b_preregistration.py
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python scripts/e01/freeze_s12b_preregistration.py
git commit ... && git push origin eidosoma/groups/42
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src python scripts/e01/run_s12b_source_audit.py
```

## Dependencies and precision

Python `3.13.14`, NumPy `2.4.6`, SciPy `1.18.0`, NetworkX `3.6.1`, pandas `2.3.3`, and PyArrow `24.0.0` were used. BLAS/OpenMP thread counts were one. Six source-analysis workers were reserved by design, but no trajectory benchmark or worker pool was launched. No GPU computation was used.

## Source equivalence validation

Seven of eight source-equivalence rows passed. Both coupled-Gaussian fixtures passed for both implementations: maximum MI difference was `9.020562075079397e-17` (gate `1e-10`), maximum partition-average difference was exactly zero (gate `1e-10`), and maximum local Phi-r difference was `8.881784197001252e-16` (gate `1e-9`). All eight untouched-source and wrapper replays were exact. Both constant-input statuses matched, and the PhiRL singular-input status matched. The sole failure was IIGR on `SINGULAR_DUPLICATE_INPUT`: the untouched pinned source returned `ELIGIBLE`, while the clean-room wrapper returned `INELIGIBLE_SOURCE_PIPELINE_EXCEPTION`; its retained-set and partition comparisons were consequently unavailable/nonmatching. The preregistration required identical singular status, so this one row failed the global gate. Raw pickle loading occurred only in isolated disposable equivalence/conversion processes; no GARD scientific execution occurred.

## Results

| Implementation | Full finite coverage | Full median rho | Prefix coverage | Prefix median current rho | Full coherence | Prefix candidate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| No scientific summary emitted because execution stopped at a mandatory gate. | | | | | | |

### Future-dependence audit

- Not evaluated because a stop condition fired.

### Spike and temporal diagnostics

- Not evaluated because a stop condition fired.

The scientific tables are deliberately empty, schema-bearing artifacts. Positive/negative 3-sigma, robust-MAD, aggregate-trend, Ljung–Box, full-versus-prefix, first-quarter, partition-ARI, sign, rank, replication-association, and `emergence` diagnostics were not calculated.

## Decision and interpretation

The frozen decision is **`SOURCE_RECONSTRUCTION_FAILED`**. S12 remains unchanged and is not rescued, overturned, or substituted. S13 remains `BLOCKED_PENDING_S12B_HUMAN_REVIEW`; no intervention authorization follows from this audit.

## Validation

FAIL CLOSED — `SOURCE_EQUIVALENCE_FAILED`. Seven of eight equivalence rows passed; the mandatory IIGR singular-status row failed. Source identities, isolated lattice conversion, exact replay, pre/post S10–S12 immutability, required-file presence, report-copy equality, hashes, and storage passed. Suffix, benchmark, coverage, and scientific-association gates were not reached. Runtime projection was not performed.

## Provenance

- Pre-outcome design Git commit: `0f975d66edb32d927535e4eaa72b4fd1105ece99`; remote commit: `0f975d66edb32d927535e4eaa72b4fd1105ece99`.
- Source snapshots, file SHA-256s, Git blobs/trees, regularization ancestry, safe-lattice opcode audit, and license note: `source_snapshot_manifest.json` and `source_audit.md`.
- S12 input hashes and S10–S12 pre/post aggregate identities: `immutable_input_audit.json`.
- Domain-separated preprocessing, Fiedler, bootstrap, shuffle, and suffix-test seeds derive from the frozen S12B root identity; row-level preprocessing and partition seeds are retained in the Parquet outputs.

## Caveats, blockers, failed assumptions, and limitations

- IIGR singular-fixture status mismatch: pinned source `ELIGIBLE`; clean-room wrapper `INELIGIBLE_SOURCE_PIPELINE_EXCEPTION`.
- The failed status gate prevents use of either wrapper on GARD data under the frozen all-branches-must-pass rule; it does not establish how either implementation would behave on the twelve trajectories.
- No post-outcome repair or scope reduction was attempted.
- S10–S12 and the public source snapshots remain immutable.

Full-mode local values use future observations in their fitted preprocessing, Fiedler partition, means, and covariances and are retrospective descriptions only. Prefix values begin only after 256 preceding transitions and are not the paper's fixed window or MLP experiment. The frozen delta, component drop, observation stream, explicit Fiedler RNG, and historical labels are reconstruction choices, not identified unpublished-author defaults. Public source similarity cannot establish causal control, early warning, exact Figures 2–4, or exact GARD implementation identity.

## Recommended next action

Return for human review. Do not repair after outcomes, do not begin S13, and preserve the failed source audit. Stop here for mandatory human review; do not begin S13, interventions, another repair, new simulations, MLP/RL work, gene-regulatory experiments, or BioModels downloads.
