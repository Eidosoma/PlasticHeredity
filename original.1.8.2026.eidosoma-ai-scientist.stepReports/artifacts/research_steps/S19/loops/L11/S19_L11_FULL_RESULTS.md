# S19-L11 Full Results — Failed-Closed All-Compotype Union Reconstruction

## Concise top summary

- **Research step ID:** `S19-L11` (`E01-S19-L11-ALL-COMPTYPE-UNION-LABEL-RECONSTRUCTION-v1.0.0`).
- **Completion status:** `LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW`; the locked global stop fired during initial label analysis, and no repair, rerun, scientific aggregation, bootstrap, control adjudication, or regeneration was performed.
- **Artifacts written:** complete pre-outcome audits/lock/fixtures/seed and 200-trajectory manifests; the 200-row exception ledger; explicit empty scientific result schemas; failure diagnosis, runtime/storage/immutability/status/classification/hash records; 12 clearly marked non-scientific figure placeholders; canonical full report and one-page handoff.
- **Validation result:** FAIL_CLOSED_AFTER_12_FIXTURE_FAMILIES_SEED_FIREWALL_CLEAN_PUSHED_LOCK_AND_200_TRAJECTORY_GENERATION;_200_OF_200_U2_WORKERS_REJECTED_MACHINE_SCALE_NEGATIVE_KMEANS_CENTROID_RESIDUE;_NO_SCIENTIFIC_AGGREGATION_OR_REGENERATION_ELIGIBLE.
- **Outcome classification:** `LOOP_FAILED_CLOSED`, `POSSIBLE_PIPELINE_ARTIFACT`, `NOT_PROMOTABLE`. U2 failed operationally on all 200 trajectories; U1 partial worker output is invalidated and was not serialized or interpreted.
- **Caveats or blockers:** the locked U2 union scorer passed K-means centroids through a nonnegativity validator without the L10 numerical clamp for machine-scale negative residue. Correcting that code after outcome access would be a method amendment requiring a new human decision. The scientific union-label hypothesis remains unadjudicated.
- **Lay summary:** The new simulations completed, but the label-analysis program stopped before it could produce trustworthy occupancy or timing results. Euclidean K-means created mathematically nonnegative centroids with tiny floating-point coordinates just below zero; the new union helper treated those as invalid. Because this exception was not an expected scientific status, the prospective contract required the entire loop to stop.
- **Recommended next action:** mandatory human review. Do not repair or rerun L11, activate another loop/S20/E02, contact authors, or run emergence, prediction, intervention, metric-distinctiveness, or report-bundle work automatically.

## Lay summary

L11 was designed to test whether “replicating” means belonging to any recurring compotype rather than only the single dominant one. The outcome-blind design, fixtures, new seed firewall, clean pushed commit, benchmark, and all 200 candidate trajectories succeeded. The first full label-analysis pass then found the same technical failure on every U2 trajectory: a Euclidean K-means centroid contained floating-point residue below zero (for the diagnosed first trajectory, minimum `-2.3852447794681098e-18`; 19 coordinates were negative at this scale). The locked direct-union helper rejected any negative coordinate before applying H.

The prior L10 direct-reference primitive had an explicit `-1e-12` numerical tolerance and clamped machine-scale residue to zero. L11's new multi-centroid helper failed to preserve that policy. This is strong evidence of a pipeline artifact, but changing it now would alter locked code after scientific outcome access. Therefore L11 answers no biological or paper-replication question.

## Frozen question

Would the union of all historical compotype tags (U1) or all Euclidean clusters with at least two members (U2) reproduce the paper-facing self-replicator fingerprint better than the frozen comparators? U1 was a boundary tag projected over the following molecular interval; U2 was direct molecular strict `max H>0.9` membership. No threshold, exposure, simulator, clock, clustering, emergence, prediction, or intervention search was authorized.

## Inputs and provenance

- Exactly 100 new shared catalytic matrices and matched initial states passed a zero-overlap firewall.
- Exactly 200 original-exposure trajectories were generated: 100 candidate 2 and 100 candidate 3; all completed 100 fissions, and none was replaced.
- Historical GARD commit `86dff6320d5ae91b4e831471079ff46749b14df9`, the original paper/Figure 1, and the L10 lock/technical repair were hashed.
- The outcome-blind repository contract was committed and pushed as `b280835ffa6d4ad06e18e33fbe1621e974926429`; release-gate `HEAD`, remote, clean-worktree, code, source, seed, fixture, and immutable-prior checks passed.

## Methods and commands

Before trajectory generation, all six source-tag audit checks and all 12 mandatory fixture families (15 checks) passed. The fixtures established multi-cluster union behavior, singleton handling, direct U2 membership, U1 projection, transformations, replay, and exception provenance. The 10-matrix opaque benchmark opened no label result and projected total use below ceilings.

```text
PYTHONPATH=src:. pytest -q tests/e01/test_s19_l11.py
PYTHONPATH=src:. python scripts/e01/run_s19_l11.py prepare
git commit ... && git push origin eidosoma/groups/42
PYTHONPATH=src:. python scripts/e01/run_s19_l11.py generate --workers 8
PYTHONPATH=src:. python scripts/e01/run_s19_l11.py analyze --workers 8
# analyze invoked the global stop; regenerate/finalize scientific paths were not run
```

CPU float64 was authoritative; GPU use was zero; eight workers each used one numerical-library thread.

## Failure result

`failure_ledger.csv` contains 200 registered rows spanning 100 matrices, with `{'CANDIDATE_2': 100, 'CANDIDATE_3': 100}` by candidate. Every row is U2 and records:

```text
ValueError: compositions must be nonnegative
```

The error arose when `direct_union_scores` called `close_rows` on R2 K-means centroids. The diagnosed first trajectory's selected R2 fit itself was eligible (`k=3`, cluster sizes `11/34/55`); its centroids' minimum was `-2.3852447794681098e-18`, consistent with floating-point cancellation rather than a material negative composition. This diagnosis is operational only and is not a label or fingerprint result.

Because U2 is required, the entire loop stopped. U1 calculations returned inside some workers, but process scheduling and the global stop make those partial values ineligible. They were neither collected into scientific tables nor used to infer occupancy, persistence, consistency, onset, episodes, controls, candidate agreement, or paper distance.

## Machine-readable result status

All directed scientific tables exist as explicit zero-row schemas bearing no scientific value. `classification.json` marks U2 `LOOP_FAILED_CLOSED` and U1 `NOT_EVALUATED_GLOBAL_STOP`. `regeneration_validation.json` says `NOT_RUN_GLOBAL_STOP`; it is not reported as a pass. The 12 required figure paths are explicit failed-closed placeholders and contain no plotted data.

## Validation

- Source audit: 6/6 checks passed.
- Mandatory fixtures: 12/12 families and 15/15 checks passed.
- Seed/input firewall: 100 unique matrices and 100 unique initial states; zero prior overlap.
- Generation: 200/200 trajectories retained, all with 100 fissions, zero replacements.
- Clean pushed release gate: passed at `b280835ffa6d4ad06e18e33fbe1621e974926429`.
- Immutable prior: passed after failure; S01–S18, V1/V2, L01–L10, and the S17 waiver remain unchanged.
- Scientific aggregation, 4,096 bootstraps, controls, full regeneration, and promotion gates: `NOT_RUN_GLOBAL_STOP`.

## Figures

The 12 required paths are failure placeholders only. Each states that no scientific value is plotted; none may be interpreted as zero evidence or a null result.

## Caveats, blockers, and interpretation boundary

1. This is an operational failure, not support or non-support for an all-compotype union label.
2. It provides no estimate of whether occupancy approaches 0.88 and no temporal fingerprint.
3. It cannot favor U1 over U2; U1 partial output is invalidated.
4. It cannot change L08/L09/L10, S18, prediction, or causal-control classifications.
5. The likely repair is narrow and source-consistent—reuse L10's material-negative tolerance for every centroid in the union—but it was not prospectively locked and was not applied.
6. Any repair/rerun requires a new additive human authorization; L11 itself must remain failed closed.

## Provenance

`implementation_lock.json`, `source_snapshot_manifest.json`, `seed_firewall.json`, `input_manifest.json`, `trajectory_manifest.parquet`, `failure_ledger.csv`, `analysis_failure_001.json`, `immutable_prior_validation.json`, and `artifact_manifest.json` preserve the full chain. Historical unlicensed source remains cache-only.

## Outcome and mandatory handoff

The machine-authoritative classification is `LOOP_FAILED_CLOSED`, `POSSIBLE_PIPELINE_ARTIFACT`, `NOT_PROMOTABLE`. No lead is promoted. Control returns for mandatory human review, with no downstream action active.
