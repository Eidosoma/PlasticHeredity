# E01/S19-L06 — Past-only multi-attractor boundary recurrence (failed closed)

## Concise top summary

- **Research step ID:** `S19-L06`
- **Completion status:** `LOOP_FAILED_CLOSED`; mandatory human-review boundary active
- **Artifacts written:** complete preregistration/lock/replay/failure/status/runtime/storage/hash package; explicit empty-not-eligible label, fingerprint, bootstrap, permutation, suffix and comparison tables; canonical full report and decision summary; append-only S19 handoff ledgers
- **Validation result:** `FAIL_CLOSED_INDEPENDENT_BOUNDARY_SCORE_EXACT_REPLAY`; immutable prior PASS across 1,664 files; preanalysis replay and pushed lock PASS; independent boundary score exact replay FAIL
- **Outcome classification:** `LOOP_FAILED_CLOSED`; `POSSIBLE_PIPELINE_ARTIFACT`; zero promoted leads; no scientific temporal fingerprint adjudicated
- **Caveats or blockers:** primary and independent float64 paths agreed on labels and recurrence counts but differed bitwise for 401/713 finite scores on the first diagnosed trajectory (maximum absolute difference `3.3306690738754696e-16`); the locked contract allowed no tolerance or post-outcome repair
- **Recommended next action:** mandatory human review; do not repair or rerun L06 under this authorization, and do not activate L07, S20, E02, author contact, or report generation automatically

## Lay summary

L06 cannot answer whether online recurrence among post-fission boundaries better reproduces the paper's 88% replicator-state fingerprint. The input data and frozen labels replayed exactly, and two executions of the new label agreed on every positive/negative decision and recurrence count inspected. However, the preregistration also required the underlying floating-point boundary scores to match **bit for bit**. Two mathematically equivalent calculation orders differed at machine precision—at most about three ten-quadrillionths—but that still violates the exact gate. The loop therefore stopped before any occupancy, onset, consistency, episode, bootstrap, permutation, or promotion result became eligible.

This is an operational validation failure, not evidence for or against the boundary-recurrence hypothesis. L06 preserves the failure rather than relaxing its rule after outcomes.

## Frozen question

The locked question was whether a single strict-`H>0.9`, past-only recurrence decision among multiple selected post-fission boundaries, projected prospectively through the following growth interval, jointly improved the paper-facing temporal fingerprint in candidate 2 and candidate 3. Adjacent molecular `H>0.9` was comparator-only; L03 and L05 were fixed prior evidence. Occupancy alone could not decide success.

## Inputs

- Frozen S13Y dataset: 100 shared catalytic matrices, 200 candidate-specific trajectories, 180,635 selected molecular-clock rows, and 20,000 selected post-fission boundaries.
- Candidate 2 and candidate 3 were kept separate.
- Frozen S13Y trajectory/cache identities, selected clocks, adjacent-H arrays, and `Y=I(H>0.9)` labels.
- Frozen L03 post-fission boundary identities and fixed L03/L05 comparison evidence.
- Original paper and source snapshots recorded in `source_snapshot_manifest.json`.
- No new GARD trajectory, PhiRL value, emergence value, prediction, intervention, threshold, recurrence-count variant, cluster, centroid, modal reference, or alignment branch.

## Detailed methods

At boundary `b_g` for positive generation `g`, the sole structural rule would activate iff strict historical cosine similarity exceeded 0.9 for at least one prior selected boundary `b_h` satisfying `0<h<=g-2`. Each earlier generation counted once. The decision would label `b_g` and selected molecular rows up to but excluding `b_(g+1)`, with no future reference, backfill, persistence across re-evaluation, or alternative branch.

The prospectively locked validation hierarchy was:

1. byte/hash replay of every immutable S01–S18/V1/V2/S19-L01–L05 input;
2. exact replay of all S13Y identities, clocks, boundary identities, adjacent-H arrays and frozen labels;
3. clean, pushed repository/config lock;
4. exact two-pass and independent replay of labels, scores and recurrence evidence;
5. only after those checks, fingerprint, suffix, bootstrap, leave-one-out, permutation and promotion analyses.

The failure occurred at item 4, so items 5 and all scientific adjudication are ineligible.

## Commands and execution chronology

Pre-outcome tests and preparation:

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l03.py tests/e01/test_s19_l04.py tests/e01/test_s19_l05.py tests/e01/test_s19_l06.py
PYTHONPATH=src ruff check src/e01_s19_boundary_recurrence scripts/e01/prepare_s19_l06_lock.py scripts/e01/run_s19_l06.py tests/e01/test_s19_l06.py
PYTHONPATH=src python scripts/e01/prepare_s19_l06_lock.py
```

The original scientific launch failed at module import before outcome access. `S19-L06-VPA-001` added only the repository root to `sys.path`, preserved that failure, reran tests, committed, pushed, and revalidated the clean lock. It changed no scientific rule or value.

Locked execution:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l06.py --workers 8
```

All workers returned, but the runner raised `L06 trajectory execution or locked audit failed` before aggregation and artifact serialization. A validation-only diagnosis of the first failing trajectory established the exact failure signature recorded below. No scientific repair, tolerance, aggregation, or rerun followed.

## Results

### Eligible scientific results

None. All scientific result tables have explicit schemas and zero rows. Invalidated worker caches remain disposable under `/cache`; they were not read or serialized by this finalizer.

| Item | Result | Eligibility |
|---|---:|---|
| Frozen preanalysis trajectories | 200/200 passed | Validation evidence only |
| Frozen selected clock rows | 180,635 replayed | Validation evidence only |
| Frozen post-fission boundaries | 20,000 replayed | Validation evidence only |
| First diagnosed independent-score trajectory | candidate 2, M000 | Validation evidence only |
| Finite boundary scores compared | 713 | Validation evidence only |
| Bitwise unequal scores | 401 | **Global stop** |
| Maximum absolute score difference | `3.3306690738754696e-16` | **Global stop** |
| Label mismatches | 0 | Does not override failed score gate |
| Recurrence-count mismatches | 0 | Does not override failed score gate |
| Eligible fingerprints / promotions | 0 / 0 | No adjudication |

No occupancy, persistence, onset, consistency, episode, quarter-cutoff, recurrence, cross-candidate, joint-distance, bootstrap, leave-one-out, block-permutation, retrospective resemblance, prediction, intervention, or causal-control conclusion is drawn.

## Validation

- Immutable S01–S18/V1/V2/S19-L01–L05 postcheck: **PASS**, 1,664 files and zero mismatches.
- Preanalysis replay: **PASS**, all 200 trajectories and all frozen identity/clock/boundary/H/label fields.
- Clean pushed pre-outcome lock: **PASS** at `0c4460ce6db913c98cbc4a898af47fe4afe54b12`.
- Pre-outcome launch-path amendment: **PASS**, value-preserving.
- Primary two-pass label/score/boundary replay on the diagnosed trajectory: **PASS**.
- Independent labels and recurrence counts on that trajectory: **PASS**.
- Independent bit-exact boundary scores: **FAIL**.
- Whole-loop eligibility: **FAIL CLOSED**.
- New trajectories / PhiRL / emergence / GPU use: 0 / 0 / 0 / 0.
- Retained-artifact storage: PASS.

## Self-improvement record

- **Belief before:** boundary-only multi-attractor recurrence might occupy the middle ground between L03's restrictive modal compotype and L05's permissive molecular recurrence.
- **Evidence motivating the test:** L03 and L05 bracketed temporal fingerprints under distinct recurrence granularities.
- **Ambiguity targeted:** whether generation-boundary granularity filters local compositional drift and creates meaningful online pre-onset intervals.
- **What was learned:** differently ordered float64 cosine implementations cannot satisfy the locked bit-exact score replay even when their label decisions and counts coincide.
- **Hypothesis weakened:** bit-exact numerical identity of those two implementation paths.
- **Hypothesis still plausible:** the boundary-recurrence scientific hypothesis remains untested by eligible L06 evidence.
- **What should be tested next:** nothing automatically. A human must decide whether any new prospectively locked action is warranted.
- **Why a future action could add information:** it could define numerical replay semantics before new outcomes; retroactively tolerating L06 would instead weaken the preregistered gate.

## Caveats and blockers

The failure magnitude is compatible with ordinary float64 operation-order rounding, but that inference does not excuse the locked bit-exact requirement. Only the first failing trajectory was diagnosed because one failure is sufficient for the global stop. Candidate 3 and the complete fingerprint remain scientifically unadjudicated. The known paper fingerprint and five previous adaptive loops also make any eventual positive label result exploratory until untouched confirmation. Exact author semantics remain unavailable.

## Provenance

The original pre-outcome lock was pushed at `1a1f7f61582c2d39c1b1df241cb029131e2326b6`; the value-preserving launch amendment was pushed and clean at `0c4460ce6db913c98cbc4a898af47fe4afe54b12`. Contracts, sources, inputs, seeds, benchmark and replay evidence are named in the L06 directory. `reporting_amendment_002.json` and this repository finalizer document the post-stop reporting-only action. `artifact_manifest.json` hashes the final retained package.

## Mandatory human-review boundary

L06 is complete as `LOOP_FAILED_CLOSED`. No L07, S20, E02, author contact, or report-bundle work is active. Hand control back for a new explicit human decision.
