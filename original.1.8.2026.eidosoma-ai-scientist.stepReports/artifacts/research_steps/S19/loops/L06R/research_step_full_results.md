# E01/S19-L06R Full Results — Numerical-Equivalence Confirmation

## Concise handoff summary

- **Research step ID:** `S19-L06R`
- **Completion status:** `LOOP_FAILED_CLOSED_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Artifacts written:** pushed repair preregistration/method lock; immutable-prior and frozen-input replay evidence; synthetic fixtures and benchmark; all-200-trajectory numerical-equivalence evidence; explicit ineligible result/control schemas; failure, classification, runtime, storage, regeneration, status, provenance and hash manifests; canonical full report and decision summary; append-only S19 ledgers
- **Validation result:** `FAIL_CLOSED_EXACT_RECURRENCE_COUNT_AND_MATCH_IDENTITY_REPLAY`; 198/200 trajectory replays passed the complete contract, while one trajectory in each candidate failed exact recurrence-count and matching-generation identity despite exact boolean labels and passing every score tolerance
- **Outcome classification:** `LOOP_FAILED_CLOSED`; `POSSIBLE_PIPELINE_ARTIFACT`; `NOT_PROMOTABLE`; constraining operational validation failure; no scientific fingerprint was released
- **Caveats or blockers:** this is an adaptive post-failure repair; failed L06 remains immutable; the score policy passed but the exact discrete gate failed; no second repair is permitted; exact author semantics remain unavailable
- **Recommended next action:** mandatory human review. Do not activate L07, S20, E02, author contact, or report generation automatically.

## Lay summary

L06R tested whether L06 had been stopped merely because two mathematically equivalent floating-point calculations differed in their final bits. Across all 200 frozen trajectories, every finite score pair passed the new, prospectively locked numerical rule: absolute errors were below `1e-12`, relative errors were below `1e-12`, and ULP distances were at most eight. The observed maxima were far smaller—`7.771561172376096e-16`, `8.881174182022044e-16`, and 7 ULP.

That did not release the scientific analysis. The contract also required exact boolean labels, recurrence counts, and matching-generation identities. Boolean labels were exact on all 200 trajectories, but recurrence counts and matching-generation lists differed on candidate-2 matrix 29 and candidate-3 matrix 98. Thus 198 trajectories passed the whole gate and two failed it. Under the one-repair-only instruction, L06R stopped permanently before occupancy, onset, consistency, bootstrap, suffix, permutation, quarter-eligibility, or promotion results were calculated or accepted. The boundary-recurrence hypothesis remains scientifically unadjudicated.

## Frozen question

Do the unchanged L06 canonical and independent CPU-float64 boundary-score paths agree under the already documented S06 numerical-equivalence policy across all 200 frozen trajectories, thereby permitting the complete unchanged L06 analysis to be adjudicated?

This repair changed only the independent floating-score equality gate. It changed no trajectory, candidate, threshold, recurrence rule, boundary selection, projection, comparator, seed, statistic, negative control, or promotion criterion.

## Inputs

- Frozen S13Y `trajectory_manifest.parquet` and `label_values.parquet` covering 100 shared matrices under each of candidate 2 and candidate 3.
- The same 200 immutable trajectory caches used by S13Y; all cache hashes replayed before L06R outcome access.
- Immutable L06 scientific method lock, label/specification registries, seed manifest, controls, and untouched-S20 design.
- Frozen S06 trajectory precision contract, which requires simultaneous absolute, relative, and ULP bounds for corresponding finite values while keeping discrete values exact.
- Original paper and prior source context only through their already frozen L06 input/source manifests.

No new GARD trajectory, PhiRL or emergence value, prediction model, intervention, GPU computation, source search, or author contact occurred. Invalidated L06 worker caches were not used.

## Detailed methods

### Historical and pre-outcome boundary

L06 remains an immutable failed-closed historical step. L06R was added under version `E01-S19-L06R-NUMERICAL-EQUIVALENCE-CONFIRMATION-v1.0.0`. Before repaired outcomes, the exact repair contract and tests were committed and pushed at Git commit `d9243e45afa969d83d1b1a1e47dec97dd249c475` on `eidosoma/groups/42`; local and remote heads matched and the repository worktree was clean.

The immutable baseline contained 1,723 S01–S18/V1/V2/S19-L01–L06 files, including the S17 waiver. All sizes and SHA-256 values matched. Preanalysis independently replayed all 200 trajectory identities, 180,635 selected-clock rows, 20,000 post-fission boundaries, adjacent-H arrays, and frozen strict-`H>0.9` labels.

### One-repair numerical contract

The unchanged canonical path was `boundary_recurrence` and the unchanged independent scalar-loop path was `boundary_recurrence_reference`, both CPU float64. The independent comparison required:

1. identical shapes, finite/nonfinite masks, and nonfinite classes;
2. exact boolean labels;
3. exact recurrence counts, first/last matches, source-boundary generations, and matching-generation identities;
4. for every finite score pair, absolute error `<=1e-12`;
5. for every finite score pair, relative error `<=1e-12` where defined; and
6. for every finite score pair, ULP distance `<=8`.

All three floating bounds were conjunctive. An 8-ULP synthetic boundary case passed and a 9-ULP case failed. The original L06 synthetic boundary trajectory passed with a maximum distance of 3 ULP. The complete benchmark projected 3.654 CPU-hours including 3.2 CPU-hours reserved for validation/finalization, below the 32-hour ceiling.

### Staged release gate

Eight one-thread workers recomputed only the canonical and independent score/discrete paths for all 200 frozen trajectories into the fresh L06R process. The complete original L06 fingerprint, future-suffix, 4,096-bootstrap, leave-one-out, 4,096-generation-block-permutation, cross-candidate, quarter-eligibility, and promotion stages were conditionally gated on all 200 numerical/discrete checks passing. Because two checks failed, those stages were never released.

## Results

### Floating-score equivalence

| Scope | Finite pairs | Non-bit-exact pairs | Maximum absolute error | Maximum relative error | Maximum ULP | All score bounds passed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Projected molecular-clock scores | 176,095 | 120,741 | 7.771561172376096e-16 | 8.881174182022044e-16 | 7 | Yes |
| Post-fission boundary scores | 19,600 | 13,442 | 7.771561172376096e-16 | 8.881174182022044e-16 | 7 | Yes |

Finite/nonfinite masks and nonfinite classes were exact for all 200 trajectories. Every absolute, relative, and ULP comparison passed. Candidate-specific projected-score maxima were 7 ULP and `7.771561172376096e-16` absolute error in both candidates.

### Exact discrete replay failure

| Candidate | Matrix | Boolean labels | Distinct counts | Qualifying counts | First match | Last match | Source boundary | Matching-generation identities | Complete gate |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| S12F-CANDIDATE-02 | 29 | Exact | Failed | Failed | Exact | Failed | Exact | Failed | Failed |
| S12F-CANDIDATE-03 | 98 | Exact | Failed | Failed | Exact | Failed | Exact | Failed | Failed |

The other 198 trajectories passed the complete numerical-plus-discrete contract. Exact boolean labels passed for all 200, so the two failures did not flip the binary boundary-recurrence state under the two implementations. They did change which qualifying historical boundaries were counted and retained. The retained gate evidence does not establish whether this is solely threshold-adjacent rounding or another independent-path ordering detail; the instruction forbids a second repair, so no further scientific diagnosis or aggregation was performed.

### Scientific release status

No temporal fingerprint or paper-distance result is eligible. The retained scientific result, negative-control, robustness, suffix, bootstrap, leave-one-out, block-permutation, quarter-eligibility, cross-candidate, and promotion tables are explicit empty/ineligible schemas. Zero leads were promoted.

## Validation

- **Passed:** clean pushed repository lock; 1,723-file immutable baseline; exact 200-trajectory cache/identity replay; 180,635 selected-clock and 20,000 boundary replay; adjacent-H and frozen-label replay; synthetic fixtures; compute/storage ceilings; fresh-cache isolation; all score-mask/class/absolute/relative/ULP checks.
- **Failed:** exact recurrence-count, last-match, and matching-generation identity on 2/200 trajectories.
- **Global decision:** fail closed before all unchanged L06 scientific analyses. No second repair is permitted.
- **Prior preservation:** failed L06 and all S01–S18/L01–L06 evidence remain unchanged; no later result was written into a prior step.

## Commands, dependencies, and runtime

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l06r.py tests/e01/test_s19_l06.py
ruff check src/e01_s19_boundary_recurrence_repair scripts/e01/prepare_s19_l06r_lock.py scripts/e01/run_s19_l06r.py tests/e01/test_s19_l06r.py
python scripts/e01/prepare_s19_l06r_lock.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l06r.py --workers 8
```

The test suite passed 9/9 tests. CPU float64 was authoritative; GPU use was zero. The all-trajectory numerical gate used 18.505924 aggregate worker CPU-seconds (`0.0051405` CPU-hours); maximum per-worker-task wall time was 0.313499 seconds. Python/NumPy/pandas/PyArrow identities are recorded in the runtime and source manifests. The retained artifact directory and fresh temporary cache remained far below their 10 GiB and 25 GiB ceilings.

## Provenance and regeneration

`preregistration.yaml`, `method_lock.json`, and `numerical_equivalence_contract.json` define the repair. `scientific_contract_reuse_validation.json` proves the L06 scientific method was unchanged; `seed_reuse_validation.json` proves byte-identical seed-manifest reuse. `numerical_equivalence_results.parquet` contains one status-bearing row per trajectory, and `numerical_equivalence_summary.json` gives candidate and global gates. Input/source identities, runtime, failure, regeneration, storage, immutable-prior, status, and artifact manifests provide the remaining provenance.

## Caveats, blockers, and interpretation boundary

1. L06R is adaptive because the numerical criterion was introduced after observing L06's bit-exact failure, even though it was locked before L06R outcomes.
2. Passing the score tolerances does not override the exact discrete replay requirement.
3. The two discrete differences did not change boolean labels, but they violate the declared recurrence-evidence identity and therefore invalidate scientific release.
4. No occupancy value, including proximity to 88%, can be inferred from L06R because the fingerprint stage was not released.
5. This step changes no retrospective association, prospective prediction, intervention, or causal-control conclusion.
6. Exact author-code identity and author label semantics remain unavailable.

## Outcome and recommended next action

L06R is permanently `LOOP_FAILED_CLOSED`, classified as `POSSIBLE_PIPELINE_ARTIFACT` and `NOT_PROMOTABLE`. Its useful result is narrow: the independent finite scores are numerically equivalent under the S06 bounds, but exact recurrence membership evidence is not identical on two trajectories. Return control for mandatory human review. Do not activate L07, S20, E02, author contact, or report generation automatically.
