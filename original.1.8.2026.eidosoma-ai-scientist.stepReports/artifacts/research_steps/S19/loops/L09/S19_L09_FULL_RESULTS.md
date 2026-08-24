# S19-L09 Full Results — Failed-Closed Recurring-Attractor Reconstruction

## Concise top summary

- **Research step ID:** `S19-L09` (`E01-S19-L09-RECURRING-ATTRACTOR-LABEL-RECONSTRUCTION-v1.0.0`).
- **Completion status:** `LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW`; execution stopped before eligible scientific serialization.
- **Artifacts written:** all directed lock/source/fixture artifacts; explicit empty ineligible scientific tables; failure, runtime, storage, regeneration, status, classification, and hash evidence; nine clearly marked non-scientific figure placeholders; canonical full report and one-page handoff.
- **Validation result:** pre-outcome 13/13 fixtures, 1,727 immutable files, 400 frozen input-cache hashes, ten-trajectory opaque benchmark, clean pushed commit, and release gate passed. Locked R1 execution then failed because an unregistered `k=n=4` all-singleton silhouette case is undefined in the backend; no full replay or scientific aggregation is eligible.
- **Outcome classification:** `LOOP_FAILED_CLOSED`, `POSSIBLE_PIPELINE_ARTIFACT`, `NOT_PROMOTABLE`. R1 failed operationally; R2 is `NOT_EVALUATED_GLOBAL_STOP`.
- **Caveats or blockers:** the fixture suite omitted the real-domain case where non-drift filtering leaves exactly k points. Selecting a post-outcome singleton-silhouette convention would change the scientific method. The recurring-attractor hypothesis remains unadjudicated.
- **Recommended next action:** mandatory human review. Do not repair/rerun L09 or activate another loop, S20, E02, author contact, prediction, emergence, intervention, or report generation automatically.

## Lay summary

The planned test did not produce a trustworthy answer about whether the paper used a recurring-attractor label. The historical pipeline sometimes leaves very few non-drifting generation states. In one frozen trajectory it left four, and the locked search tried to cluster those four states into four singleton clusters. The selected software does not define a silhouette score for that case. Because choosing a convention after seeing the failure could change which cluster count wins, the analysis stopped and discarded every partial result.

## Frozen question

Could either of exactly two fixed dominant-recurring-composition labels jointly reproduce the paper's control fingerprints better than adjacent, boundary, projected-boundary, and high-exposure comparators?

## Inputs

The lock used exactly the 100 shared L08 matrices and four frozen trajectory groups; original-exposure candidate 2/3 were primary and `h=2.875` candidate 2/3 comparator-only. The original paper, Figure 1, Table 1, pinned historical GARD source, and cited methods 63–65 were hashed. No new trajectory, emergence value, prediction, or intervention was generated.

## Methods and lock

R1 froze historical technique-1 non-drift filtering, cosine k-means over k=1–10 with ten replicas, source-equivalent score/early-stop behavior, dominant valid compotype selection, and direct molecular strict-H>0.9 membership. R2 froze all-boundary Euclidean Lloyd k-means and the same membership threshold. Mandatory fixtures covered source smoothing, permutation/scaling, replay, planted dominant and two-attractor structure, a drifting no-cluster case, deterministic ties, and direct molecular rather than interval-projected labels. The complete code/config lock was committed and pushed at `691b328` before scientific execution.

## Commands

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l09.py
PYTHONPATH=src python scripts/e01/prepare_s19_l09_lock.py
git commit ... && git push origin eidosoma/groups/42
PYTHONPATH=src python scripts/e01/run_s19_l09.py --workers 8
```

## Failure result

The first surfaced R1 worker failure was:

```text
ValueError: Number of labels is 4. Valid values are 2 to n_samples - 1 (inclusive)
```

After non-drift filtering, that trajectory supplied four eligible boundaries and the locked search evaluated k=4. The backend cannot calculate silhouette when every point is its own cluster. The runner stopped globally. Its exception path did not serialize the candidate/matrix identity, and the unit was deliberately not reopened after the stop. In-memory computations from other workers are invalidated and absent from the artifacts.

No occupancy, persistence, consistency, onset, episode, control, bootstrap, cross-candidate, or paper-distance result is eligible. The comparator tables are also explicit empty schemas because selective continuation would drop a required pipeline.

## Validation

- 13/13 mandatory pre-outcome source/synthetic fixture checks passed.
- 1,727 immutable prior files and all 400 L08 cache hashes passed before execution.
- The ten-trajectory opaque benchmark passed the compute gate and retained no scientific values.
- Clean pushed `HEAD == origin/eidosoma/groups/42` and locked code hashes passed.
- The real-domain k=n condition was not represented in the fixtures; this is the operational validation failure.
- Full exact replay is explicitly `NOT_RUN_GLOBAL_STOP`; it is not misreported as passed.
- Required scientific tables carry explicit empty schemas, and failure provenance is machine-readable.

## Figures

The nine directed figure paths exist only as clearly labelled failure placeholders. They contain no scientific result and prevent report consumers from mistaking absent panels for zero-valued evidence:

- `figure_01_dominant_recurring_clusters.png`
- `figure_02_molecular_h_to_dominant_over_time.png`
- `figure_03_adjacent_vs_recurring_labels.png`
- `figure_04_occupancy_persistence_comparison.png`
- `figure_05_consistency_first_onset.png`
- `figure_06_episode_topology.png`
- `figure_07_negative_controls.png`
- `figure_08_cross_candidate_agreement.png`
- `figure_09_fingerprint_decision_matrix.png`

## Caveats and interpretation boundary

This failure does not support or refute a recurring-attractor label. It cannot be used to favor R2, to reinterpret L08, or to change S18's prediction/control conclusions. A future attempt would need a new human authorization and a prospective, source-grounded choice for all-singleton silhouette handling, plus a fixture that exercises it. L09 itself is immutable and cannot be repaired.

## Provenance

`source_snapshot_manifest.json`, `label_method_lock.json`, `input_manifest.json`, `preoutcome_preparation_failure_001.json`, `failure_ledger.csv`, `run_release_gate.json`, and `immutable_prior_validation.json` preserve source, code, input, preparation, execution, and stop identities. Unlicensed historical source remains cache-only.

## Recommended next action

Stop for mandatory human review. No next option is active.
