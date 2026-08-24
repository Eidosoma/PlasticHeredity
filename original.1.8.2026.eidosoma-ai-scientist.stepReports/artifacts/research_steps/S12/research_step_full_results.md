# E01 S12 strict minimal reproducible run: full results

## Top summary

| Field | Result |
| --- | --- |
| Research step | **S12**, versioned as **`E01-S12-STRICT-MRR-v1.0.0`** |
| Completion status | **Completed the bounded authorized scope and stopped.** Twelve baseline matrices and complete 100-fission trajectories were produced. The frozen feasibility gate passed, so exactly six preregistered max/control/min triplets were also completed. No S13 work, matrix-count expansion, fixed-window repair, or invalid fixed-window estimate was started. |
| Artifacts written | The canonical directory is `/artifacts/research_steps/S12/`. It contains the frozen preregistration and two pre-outcome amendments, 12 matrices, complete baseline and intervention trajectories/events, all status-bearing estimates and action scores, coverage/association/intervention/claim tables, three figures, failure and validation records, status JSON, provenance/hash manifest, completeness audit, and this report. The final manifest is the authoritative file list and checksum inventory. Repository code is retained in Git rather than copied into the artifact directory. |
| Validation result | **Overall validation did not fully pass.** Seventeen of 19 final computation checks passed; the two preserved failures were the frozen source/CPU/GPU numerical checkpoint family and complete pinned-phyid whole-trajectory atom availability. Artifact cardinality, schema/status completeness, exact regeneration, pairing, runtime, storage, immutability, and final artifact-completeness checks passed. The status JSON therefore correctly records `success: false` despite successful completion of the bounded execution. |
| Outcome classification | **Constraining/contradictory.** Strict past-only estimates were widely available after eligibility, but all preregistered post-eligibility association medians were negative, not positive; only one of 1,090 authorized treated action opportunities was separable; and source-reference numerical checks were incomplete. |
| Caveats or blockers | This is a partial forensic reconstruction, not the unavailable author implementation or MATLAB RNG. It cannot recover fixed-window/local-spike behavior, pre-eligibility or early-warning claims, the original ineligible first-quarter analysis, intervention from the beginning or after every fission, or exact Figure 6/Table 1. MMI and experimental CCS remain distinct validation branches. Ten of 48 pinned-source whole-trajectory local branches failed closed, and 14 of 36 actual-trajectory numerical checkpoints missed the joint frozen policy. |
| Lay summary | The strict method became usable for most of each simulated trajectory, typically around generation 10. It did not show the paper-directed positive relationship in the eligible late-time region. The intervention safeguard almost always judged the apparent best action indistinguishable from chance or a numerical tie, so it appropriately did nothing in 1,089 of 1,090 opportunities. The one applied action is far too sparse to support an intervention claim. |
| Recommended next action | **Do not run S13.** Close the E01 Phi-r reconstruction as restricted and underdetermined outside its valid post-eligibility estimand. If the group wishes to study alternative causal architecture or reaction coordinates, place that work in a new, separately preregistered E02 methodological scope after human review. |

## Frozen question and decision boundary

The frozen question was whether the already validated S10 strict Gaussian PhiID branch is numerically usable on 12 actual GARD trajectories after the unchanged per-trajectory boundary of 512 effective samples; how much prospective coverage remains; whether valid past-only expanding estimates are positively associated with later or continuing self-replication; and, only if a pre-outcome feasibility gate passes, whether post-eligibility max/min actions are reproducibly distinguishable from a state-matched full-candidate null.

The evidence class was frozen as `PARTIAL_FORENSIC_REPLICATION_AND_FEASIBILITY_TEST`. The following boundary remained binding throughout:

- S10, S11, and S11R source files and artifacts remained immutable. No S11 or S11R fixed-window value was read as a scientific estimate, repaired, approximated, or relabeled.
- S10's strict `n_eff >= 512` gate was unchanged. Histories were never pooled between trajectories.
- Exactly 12 baseline matrices were authorized. The only positive intervention count allowed was exactly six triplets selected by the frozen rule; otherwise the count had to be zero.
- Neither an estimator-repair cycle nor S13, a 100-matrix run, broader intervention work, an author-primary designation, or a paper-primary designation was authorized.

The exact frozen scope statement is:

> `E01-S12-STRICT-MRR-v1.0.0 cannot recover the failed fixed-window scope and is not an exact author-implementation, Figure 6, or Table 1 replication.`

## Lay summary

The strict estimator was computationally feasible on the GARD runs after enough past observations accumulated. Depending on the trajectory, the first usable post-fission estimate appeared between generations 5 and 12, with a median of 10. Once available, the estimate covered about 94.64% of all materialized observations in each named preprocessing/redundancy branch and every trajectory was eligible at its end.

That broad late-time coverage did not restore the paper's local-time analysis. All six unique preprocessing-by-estimand summaries had negative median associations. The most decisive dropped-CLR summaries had bootstrap intervals entirely below zero; the ILR branch was also negative but two intervals included zero. These are restricted post-eligibility findings only. They say nothing about early or fixed-window behavior.

The intervention feasibility gate passed, so the first six qualifying matrices were run as max/control/min triplets. Candidate scoring itself was reproducible, but the multiple-candidate null and tie rules suppressed almost every proposed intervention. Only one max deletion was applied across 1,090 treated post-origin opportunities. That single divergence cannot sustain an intervention conclusion; all affected intervention claims remain `UNDERDETERMINED` by the frozen rule.

Finally, the direct strict calculation remained available, but the pinned phyid source failed on some long actual-trajectory local-density decompositions, and several external checkpoint differences exceeded the unusually tight `1e-10` policy. Those failures are retained rather than regularized, imputed, or hidden. They constrain the reconstruction and are why the final machine-readable status has `success: false`.

## Preregistration and outcome firewall

The preregistration was created, validated, committed, and pushed before any GARD scientific outcome was generated or inspected.

| Frozen item | Git commit | Artifact SHA-256 |
| --- | --- | --- |
| Base `E01-S12-STRICT-MRR-v1.0.0` | `9b896503a921d3b47e606ef772e9fcedabd75a5b` | `bb0f2c9863ebc866242c12b46117c0442ab8b2a0694eeebc5de7f5f8c2f8c99a` |
| Amendment 01: source checkpoints, action observation, tie, and endpoint semantics | `ebca2b36d601620a60e22bc661df72a5e57a0a6e` | `bfa470d1d8fe51cf48ba4aba191b3c54a933fa3cb5a6c38c6524fae87060db76` |
| Amendment 02: whole-trajectory descriptive rules, exact claim boundaries, and action-density rule | `0230069871d40f0831ef5961b6c08094bbe08a35` | `df4f25b527c134ba7d7bac566b6168abeac439558f004daeac0afa39deb0c6b6` |

Both amendments were operational clarifications made before outcomes. They changed no matrix count, strict sample threshold, S10/S11/S11R evidence, estimator identity, scientific success gate, or permitted scope. The base freeze record and amendment-02 record have SHA-256 values `b8d350609d48333c15747038dbf3211ba654df9e6362d46e82b3c1252eaf3e62` and `879cd2fe6b930da7ac2dea4c3eebccf309dee7f9aeb1a4abe2f70ce62ba15f70`, respectively.

The pre- and post-run audit passed all frozen-input, scope, and immutability checks. In particular:

| Preserved input set | File count | Aggregate SHA-256 before and after S12 |
| --- | ---: | --- |
| S10 artifacts | 21 | `aeb44f0348370c78b46cc75f33f4e20acd2f3dfd2b68b91de1013b5052593f6a` |
| S11 artifacts | 33 | `b328f9bed5458d4513891c85e4c57a8de4a0590ffb505be6c88957b2d4e90f0b` |
| S11R artifacts | 25 | `29603ad75f2d62c5547271c80ae870553c3f479262f67376eb333adc9bc4e557` |
| S10/S11/S11R tracked repository selection | 23 | `979039195c3d94ecbc18d992bcd533b1ff378cc912a1669bee4a241a310f5d70` |

## Inputs

The complete list of 28 frozen inputs, their paths, and their SHA-256 identities is in `preregistration.yaml`. Principal inputs were:

- `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and the pre-outcome `/workspace/RESEARCH_PLAN.md` (frozen plan SHA-256 `95859e7479be997c27409455d2e730db162cd3445ca241ff57523b9e6d804e61`).
- `/workspace/input-attachments/MANIFEST.json`, its `_metadata/ATTACHMENT.md` sidecar, and the supplied paper extraction. The official arXiv v1 PDF had SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- The S09 preprocessing report/contract/eligible-transform registry; the complete S10 information-dynamics report, contract, and eligibility registry; and the S11/S11R reports, contracts, eligibility registries, failure ledgers, and artifacts.
- Specification registry v0.3.0, S03 source/environment/precision manifests, historical and independent GARD engine contracts, and the S06 seed and trajectory contracts.
- Historical GARD commit `86dff6320d5ae91b4e831471079ff46749b14df9`, pinned phyid commit `6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44`, and optional OmegaID commit `7fcf1fa8e288e0634f81423283d2b349ed88440e`.

The historical GARD checkout has no detected root license, so the artifacts record commit/file identities and results without redistributing that source tree.

## Methods

### GARD generation and molecular sampling

The independently implemented, S07-validated engine was configured to one explicit source-traceable public-historical branch: 100 species, `n_min=40`, `n_max=80`, 100 fissions, `A=-4`, `sigma=4`, `k_f=0.01`, `k_b=0.0001`, uniform reservoir concentration, historical catalytic-matrix orientation with diagonal, categorical one-event updates, eventwise exact growth stopping, fixed-size sampling without replacement at fission with odd discard, first-daughter continuation, and with-replacement initialization. The historical unbounded growth-loop interpretation was used; the paper-prose `max_steps`, Poisson, binomial-fission, and distinct-initialization conflicts remain explicit and were not silently reconciled.

Integer states were retained at the initial state, immediately after every `+1/-1` molecular event, and immediately after every fission and first-daughter selection. A fission boundary was a real adjacent transition and was not deleted or smoothed. `observationIndex` is zero-based over this complete sequence, `molecularStep` counts only molecular events, and `generation` counts completed fissions.

The sole prospective lag was `tau=1` adjacent materialized transition. At observation index `t`, the prefix contained only observations `0..t`; hence `n_eff(t)=t`. A trajectory became sample-eligible only when its own `n_eff >= 512`. No future observation, completed-trajectory covariance, retrospective partition, or cross-trajectory pooling was allowed in a prospective value.

### Preprocessing

Every integer state was processed under both frozen, validation-only 99-dimensional representations:

1. Additive pseudocount `delta=0.5`, closure, full CLR, then drop original component 100.
2. The same `delta=0.5` and closure followed by a 99-dimensional orthonormal Helmert ILR transform.

Every input row was retained. Each transform required finite output and an inverse/forward maximum absolute error no greater than `1e-10`. Neither branch is claimed as the recovered paper default.

### Partition and strict Phi-r estimate

For each trajectory and preprocessing branch, the partition search was attempted at post-fission points after the sample threshold. It built an absolute-correlation adjacency over all 99 coordinate series, used the normalized-Laplacian Fiedler vector with a deterministic orientation, split by sign, and mapped each part to its coordinate-wise arithmetic mean. The first candidate satisfying the frozen finite, eigengap, nonempty-side, minimum-side-fraction `0.1`, finite-objective, exact replay, and three-permutation feature-relabel checks was locked using past-only data and never revised. Until then, rows were emitted as `PARTITION_NOT_YET_LOCKED`.

The strict scalar was the equation-derived aggregate

`str + stx + sty + sts - rtr - rtx - rty - rts`,

equivalently checked as `I(X_t,Y_t;X_t+1,Y_t+1) - I(X_t;X_t+1,Y_t+1) - I(Y_t;X_t+1,Y_t+1)` for the two mapped scalar series. It used an unregularized binary64 sample covariance with `ddof=1`. MMI and experimental CCS were emitted with separate IDs and provenance. Their scalar is algebraically redundancy-invariant; that does not identify the authors' atom, redundancy function, estimator, MIB objective, normalization, search, or partition mapping. CCS remains experimental because the pinned source documents that path as incomplete.

Each numeric row required `n_eff>=512`, valid preprocessing, a past-only locked partition, finite/nonconstant scalar series, positive-definite rank-4 covariance, condition number at most `1e12`, at least 20 effective samples per joint dimension, lattice closure and direct/atom agreement at most `1e-10`, exact same-engine replay, and exact specification identities. A failed row retained a null `value`, status, and reason. No regularization, fallback, imputation, or silent omission was permitted.

### Labels and association analyses

The source-traceable retrospective historical `H>0.9` non-drift label and a separately identified past-only cosine companion were materialized at all 1,200 baseline post-fission generations. The historical label is not author-code identity and uses future observations by construction.

Three restricted post-eligibility estimands were preregistered:

- strict Phi-r versus historical label at the same eligible generation (`continuing_replication`);
- strict Phi-r versus the historical label one generation later (`later_replication_one_generation`);
- strict Phi-r versus the past-only cosine companion at the same generation (`past_only_companion_continuing`).

The statistic was Spearman correlation within each trajectory followed by the median across trajectories. Uncertainty used 4,096 trajectory bootstrap replicates. The positive-direction null used 4,096 independent nonzero circular generation shifts within trajectory. MMI and CCS used separately derived estimator streams even though their scalar values coincide.

### Whole-trajectory branch

Only after a trajectory completed, a whole-trajectory partition and scalar were calculated. Every output was labeled exactly `DESCRIPTIVE_NONPROSPECTIVE`. These values were forbidden for prospective prediction, lead time, action selection, or causal-control claims. Pinned-source local atom time series were separately attempted for descriptive trend, spike, temporal-dependence, and label-association diagnostics; any source failure remained status-bearing.

### Conditional intervention design

The all-or-none baseline gate required all 12 complete trajectories, exact regeneration for matrices 0/5/11, at least six qualifying trajectories with at least 24 eligible post-fission points in both preprocessing branches, final coverage at least `0.75`, median first eligible generation at most 25, at least six full-candidate pilots with exact replay, and frozen CPU/wall/storage projections within their ceilings. A pass selected the first six ascending matrix indices among candidate-pilot-passing qualifiers.

At each authorized treated post-fission point, the complete candidate set contained no-op, `+1` for every species, and `-1` for every currently present species. A candidate was represented as one hypothetical next observation, with the existing locked partition and complete past history. Every candidate had to pass both preprocessing branches and exact replay. Max/min required the same unique candidate in both representations. Any numerical tie within `1e-12` was suppressed without using molecule index or array order.

For each direction and preprocessing branch, score residuals were centered within addition and deletion classes. A state-matched null generated 4,096 complete candidate families by resampling within each class, retained centered no-op, and used the `0.99` higher quantile of the selected top-two/bottom-two full-family gap. An intervention was permitted only when the observed best-versus-runner-up gap was strictly greater than `max(1e-10, exact replay error) + null envelope` in both preprocessing branches. Otherwise no-op was applied and the exact status `INELIGIBLE_ACTION_NOT_SEPARABLE` was emitted.

Matrix, initial-state, event, waiting-time, fission, and daughter streams were common within a triplet wherever mathematically valid. Intervention, estimator, and machine-learning streams were condition-specific. Divergence points were recorded explicitly.

### Seeds, runtime, and storage

Nine purposes used the S06 SHA-256 domain-separated PCG64DXSM contract rooted at `1212...1212` (32 bytes). Baselines were trajectory-isolated. The run used eight worker processes, one thread per numerical library, CPU float64 as the scientific backend, and one visible NVIDIA L4 for guarded OmegaID Gaussian float64 cross-checks. TF32 and mixed precision were disabled. The compact-artifact plus cache ceiling was 20 GiB; large temporary material was restricted to `/cache/e01_s12`.

## Commands and dependencies

No dependency was installed or upgraded in S12. The pinned S10 Python 3.13 environment was reused. Principal versions were Python 3.13.14, NumPy 2.4.6, SciPy 1.18.0, CuPy 13.6.0, pinned phyid source commit `6c5f2e9...`, and OmegaID 0.2.5/commit `7fcf1fa...` for guarded Gaussian cross-checks only.

The primary bounded execution command was:

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src /cache/e01_s10/venv/bin/python \
scripts/e01/run_s12_strict_mrr.py --workers 8
```

Validation and finalization commands included:

```bash
PYTHONPATH=src /cache/e01_s10/venv/bin/python -m pytest -q \
  tests/e01/test_s12_strict_mrr_preregistration.py \
  tests/e01/test_s12_strict_mrr.py

/cache/e01_s10/venv/bin/python -m ruff check \
  src/e01_strict_mrr scripts/e01/run_s12_strict_mrr.py \
  scripts/e01/finalize_s12_artifacts.py tests/e01/test_s12_strict_mrr.py \
  tests/e01/test_s12_strict_mrr_preregistration.py

CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src /cache/e01_s10/venv/bin/python \
scripts/e01/finalize_s12_artifacts.py --require-report
```

The focused suite passed 17/17 before the required completed-step edit to `RESEARCH_PLAN.md`, and targeted Ruff checks passed. After that required edit, 16/17 focused tests passed and the sole expected failure was the deliberate preregistration assertion that the live plan still equal its pre-outcome frozen hash; the immutable-input audit independently confirms that the frozen hash passed both before and after the scientific run. A broader pre-handoff run passed 92 tests and retained three analogous expected legacy state/hash guardrail failures: one S06 exact-identity check after the repository commit changed and two S09/S10 tests that compare the live, subsequently updated plan with earlier frozen hashes. These did not indicate S12 numerical or scientific failures and were not bypassed.

## Results

### Baseline trajectories and status completeness

All 12 preregistered matrices completed exactly 100 fissions.

| Payload | Rows |
| --- | ---: |
| Baseline observations | 125,690 |
| Baseline molecular-event/fission log | 125,678 |
| Preprocessing status rows | 251,380 |
| Replicator-label rows | 2,400 |
| Prospective expanding status rows | 502,760 |
| Post-fission estimate rows | 4,800 |
| Baseline partition-history rows | 234 |

Each observation has exactly four estimate rows: two preprocessing IDs by two redundancy IDs. For each such branch, 118,947 of 125,690 rows (94.6352%) were numeric and strict-eligible, 6,144 were explicitly ineligible for insufficient effective samples, and 599 were explicitly ineligible while the first qualifying past-only partition had not yet locked. Every final observation was eligible in both preprocessing branches. No value exists in an ineligible row.

The retrospective historical label marked 426/1,200 generations as replicator; the past-only cosine companion marked 461/1,200. These are distinct label branches, not interchangeable truths.

### Eligibility timing and coverage

Eligibility timing was identical for dropped CLR and ILR in this run:

| Matrix | First eligible molecular step | First eligible generation/post-fission point | Eligible post-fission points of 100 |
| ---: | ---: | ---: | ---: |
| 0 | 550 | 10 | 91 |
| 1 | 516 | 10 | 91 |
| 2 | 528 | 11 | 90 |
| 3 | 504 | 11 | 90 |
| 4 | 514 | 11 | 90 |
| 5 | 522 | 8 | 93 |
| 6 | 530 | 11 | 90 |
| 7 | 504 | 10 | 91 |
| 8 | 510 | 12 | 89 |
| 9 | 572 | 9 | 92 |
| 10 | 672 | 9 | 92 |
| 11 | 704 | 5 | 96 |

The median first eligible generation was 10, the range was 5–12, and the median number of eligible post-fission points was 91. Generation-indexed coverage reached `C(g)=1.0` by generation 12 and remained there through generation 100; every trajectory's own final observation was eligible. The molecular-step table follows the required fixed denominator `R=12`. Its `observedTrajectoryCount` therefore falls after shorter trajectories have completed, and `C(t)` correspondingly declines to `1/12` at the longest matrix-11 steps even though no still-observed trajectory lost eligibility. This is termination support, not pooled estimation or an eligibility reversal. The differing molecular-step/generation ordering for matrices 10 and 11 reflects their much longer growth segments.

### Restricted prospective associations

The table reports the separately generated MMI and CCS permutation p-values; the equation-derived scalar, medians, and bootstrap intervals coincide across those redundancy identities.

| Preprocessing | Restricted estimand | Median trajectory Spearman | Positive/defined trajectories | 95% trajectory-bootstrap interval | MMI / CCS positive-direction permutation p |
| --- | --- | ---: | ---: | --- | --- |
| Dropped CLR | Continuing historical replication | -0.086945 | 1/11 | [-0.188506, -0.046442] | 0.883329 / 0.885038 |
| Dropped CLR | Historical replication one generation later | -0.069467 | 2/11 | [-0.215902, -0.036732] | 0.833049 / 0.842568 |
| Dropped CLR | Past-only companion, continuing | -0.134075 | 2/11 | [-0.275917, -0.100099] | 0.964120 / 0.964852 |
| Helmert ILR | Continuing historical replication | -0.038732 | 2/11 | [-0.104278, -0.020064] | 0.655113 / 0.651208 |
| Helmert ILR | Historical replication one generation later | -0.055839 | 5/11 | [-0.132350, 0.031301] | 0.727606 / 0.731267 |
| Helmert ILR | Past-only companion, continuing | -0.095190 | 4/11 | [-0.120687, 0.070276] | 0.896754 / 0.893581 |

No branch approached the preregistered positive-direction gate of at least 9/12 positive trajectory statistics, a bootstrap interval excluding zero in the positive direction, and permutation `p<=0.05`. All point summaries were negative. The restricted association result is therefore not supportive; it must not be generalized backward to the first 512 transitions or to any fixed-window/local-spike estimand.

### Whole-trajectory descriptive branch

All 48 direct whole-trajectory scalars (12 matrices by two preprocessing by two redundancy identities) were finite and labeled exactly `DESCRIPTIVE_NONPROSPECTIVE`. They were never used for prospective prediction, action selection, or lead-time claims.

Pinned phyid local atom decomposition was available for only 38/48 trajectory-branch combinations. Dropped CLR failed for matrices 6, 10, and 11 under both MMI and CCS; ILR failed for matrices 10 and 11 under both. The retained reason is `PINNED_PHYID_SOURCE_FAILURE::InformationBackendError::Backend returned a nonfinite decomposition.` No CCS atom was imputed and no source series was replaced by the direct scalar.

Consequently, the four preregistered 12-trajectory aggregate trend summaries, four three-standard-deviation spike summaries, and four complete-sample temporal-dependence summaries are explicitly ineligible. Partial descriptive local-label summaries remain available for 9 dropped-CLR and 10 ILR trajectories, with median run associations of -0.031283 and +0.011710, respectively, but these incomplete, nonprospective diagnostics do not establish a directional paper comparison. The whole-trajectory branch therefore does not recover a Figure 6/local-spike result.

### Baseline-feasibility gate

Every frozen sub-gate passed. All 12 trajectories qualified and all 12 complete candidate pilots passed strict eligibility, both preprocessing branches, null construction, and exact replay. Final coverage was 1.0, the median first eligible generation was 10, baseline CPU was 0.119112 hours, median complete pilot time was 0.852194 seconds, projected complete-S12 CPU was 0.377137 hours, and projected wall time was 0.125883 hours. Storage/free-space projections passed. The frozen selection rule therefore selected exactly matrix indices 0–5, and exactly six triplets (18 conditions) were run.

### Intervention feasibility and discriminability

The intervention branch produced 94,215 status-bearing observations, 94,197 event/fission records, 376,860 expanding estimate rows, 263,794 candidate-score rows, and 1,800 post-fission action rows. Every candidate row was strict-eligible, and candidate replay error was exactly zero.

The 600 control decisions were no-op by design. Across max and min treatments, 110 pre-origin decisions were retained as `INELIGIBLE_PRE_COMMON_RISK_ORIGIN`. Of 1,090 post-origin treated opportunities:

- exactly one max action was applied (`delete:35`, matrix 2, generation 66);
- 516 max opportunities were suppressed because multiple candidates were within the frozen numerical tie tolerance;
- 28 additional max and all 545 min opportunities were suppressed because the best/runner-up gap did not exceed the numerical-plus-full-set-null envelope.

Thus 1,089/1,090 opportunities (99.9083%) emitted exactly `INELIGIBLE_ACTION_NOT_SEPARABLE`, and only 0.0917% applied an action. The frozen intervention success rule required at least four of six triplets to apply at least three separable actions in each noncontrol condition; it failed decisively.

The pairing audit passed all six triplets: catalytic matrix and initial state were common, control reproduced its corresponding baseline exactly, all conditions were state-identical through the frozen common-risk origin, and no action occurred before that origin. The sole action occurred at decision observation 3,266; the max state first diverged at observation 3,267 and raw event draws ceased to be semantically aligned at the first next-generation growth draw. No other treatment diverged.

Restricted historical-label counts after the common-risk origin were:

| Matrix | Max | Control | Min |
| ---: | ---: | ---: | ---: |
| 0 | 7 | 7 | 7 |
| 1 | 10 | 10 | 10 |
| 2 | 14 | 19 | 19 |
| 3 | 28 | 28 | 28 |
| 4 | 18 | 18 | 18 |
| 5 | 30 | 30 | 30 |

The max-minus-control mean difference was -0.8333 labels (median 0; 95% paired bootstrap interval [-2.5, 0]; exact one-sided sign-flip `p=1.0`). Control-minus-min was exactly zero for all six pairs (interval [0, 0], `p=1.0`). These contrasts are not valid intervention nulls in the broader scientific sense because almost no treatment was administered. Under the frozen action-density rule, affected intervention claims are `UNDERDETERMINED`, not `NOT_SUPPORTED_WITHIN_STRICT_SCOPE`.

### Paper-claim status matrix

After restoring the preregistered sparse-action boundary in derived metadata, all 59 claim rows use only the required vocabulary:

| Status | Count |
| --- | ---: |
| `SUPPORTED` | 0 |
| `DIRECTIONALLY_SUPPORTED` | 0 |
| `NOT_SUPPORTED_WITHIN_STRICT_SCOPE` | 7 |
| `UNDERDETERMINED` | 40 |
| `NOT_EVALUATED` | 12 |

The 40 `UNDERDETERMINED` rows include all unavailable fixed-window/local-spike timing claims; all pre-eligibility and early-warning origins before eligibility; the original first-25%-to-final-75% claim where its input is not independently eligible; interventions before eligibility or after every fission from the beginning; exact Figure 6/Table 1 reconstruction; source-incomplete whole-trajectory claims; and sparse-intervention claims. The seven `NOT_SUPPORTED_WITHIN_STRICT_SCOPE` classifications refer only to valid restricted post-eligibility estimands and are not nonreplications of early/fixed-window behavior.

## Validation

### Checks that passed

- Base preregistration and both amendments validated before outcomes; all 28 frozen inputs matched.
- S10/S11/S11R artifacts and selected repository source remained byte-for-byte identical before and after the run.
- Exactly 12 finite 100x100 catalytic matrices and twelve 100-fission baselines were present; all observation, event, preprocessing, label, and four-row-per-observation estimate cardinalities matched.
- Every ineligible estimate retained a null numeric field and explicit reason; every eligible value was finite and met the per-estimate strict gate.
- Both preprocessing branches retained all rows and passed finite inverse checks.
- Same-engine regeneration for matrices 0, 5, and 11 reproduced matrices, all states, and trajectory SHA-256 values exactly.
- All nine S06 stream purposes were present and domain-separated; 828 seed rows were recorded.
- The all-or-none intervention count, full candidate cardinality, exact candidate replay, exact suppression token, and all six pairing audits passed.
- All 48 direct whole-trajectory scalar outputs were finite and had the exact `DESCRIPTIVE_NONPROSPECTIVE` label.
- Runtime and storage ceilings passed; no bytecode, compiled objects, cache tree, dependency tree, or build product appears under the artifact root.
- The pre-plan-update focused code suite passed 17/17 and targeted Ruff checks passed; the final post-plan-update rerun passed 16/17 with only the deliberate frozen-plan-hash assertion failing because this handoff updated the live plan.
- Visual inspection confirmed that all three PNGs are legible and consistent with their tables; in particular, the molecular-step coverage decline correctly reflects the fixed `R=12` denominator after shorter trajectories terminate, while generation coverage stays at 1.0 after generation 12.
- Final artifact presence, nonempty-file, schema/cardinality, claim-vocabulary, action-boundary, manifest, and report checks passed.

### Preserved validation failures

Two of 19 final validation families failed and were not weakened:

1. **Actual-trajectory source/CPU/GPU checkpoints:** 22/36 passed and 14 failed. Eight failures were pinned-phyid nonfinite source decompositions. Six additional comparisons exceeded the joint `1e-10` absolute/relative policy. Maximum running-direct versus phyid absolute error was `6.2952e-10`; maximum phyid versus OmegaID CPU error was `6.5495e-9`; maximum OmegaID CPU versus L4 GPU error was `1.95435e-8`. The arithmetic checkpoint expectation was corrected from 72 to 36 because the frozen set is `3 matrices x 2 preprocessing branches x 3 points x 2 redundancies`; no scientific value changed.
2. **Whole pinned-source atom completeness:** 38/48 branch combinations were available and 10/48 failed closed through nonfinite local-density decomposition. Whole direct scalars were not substituted for missing source atoms.

The unit-level synthetic CPU/GPU checkpoint remained accurate (maximum absolute difference `6.217e-15`), but it cannot override failures on the actual GARD trajectories. Guarded OmegaID Gaussian was only a cross-check; OmegaID discrete and doublet substitutes were never used.

### Preserved failed execution attempt

Execution attempt 001 at implementation commit `66607261f5e8bfaa64549cf22ca159af7c1def77` stopped during completed-trajectory pinned-phyid analysis when the source returned a nonfinite decomposition. It wrote no GARD scientific outcome artifacts before failing. The failure record was retained as `execution_attempt_001_failure.json` (SHA-256 `c65aee3480a850eff2ef7a25c099f0ae924c594e669462cdf03b6cd3e498841c`).

The bounded repair at commit `91017136c3d91120e73af566616736ae817c0dc8` did not change the estimator, threshold, branch, matrix count, or outcome gate. It caught the pinned-source exception, retained the affected branch as ineligible with its exact reason, and allowed the already frozen direct strict scalar to remain separate. The full command then completed once. Two later derived-metadata fixes corrected only the checkpoint cardinality label and restored sparse intervention claims to the frozen `UNDERDETERMINED` classification. They did not rerun a simulation or change a scientific number.

## Runtime, hardware, and storage

The complete invocation took 1.06758 wall hours. Measured task CPU was 0.48347 hours: 96.873 simulation seconds, 281.606 analysis seconds, 50.324 regeneration seconds, and 1,311.673 intervention seconds. This was far below the S12 200 CPU-hour, E01 250 CPU-hour, and S12 48 wall-hour ceilings.

The runtime was Linux/glibc, Python 3.13.14, NumPy 2.4.6, SciPy 1.18.0, eight processes, and one numerical-library thread per process. The visible accelerator was NVIDIA L4 UUID `GPU-1f5bed19-d63f-be98-470c-f670e3d4aafd`, driver 610.43.02, CUDA runtime 12.9 as reported by CuPy, float64, TF32 disabled, and mixed precision disabled.

The pre-final-metadata S12 payload occupied 523,461,854 bytes. Finalization added only the report and compact metadata, leaving the measured payload at about 524 MB with zero retained bytes in `/cache/e01_s12`. The exact final byte count is recorded in `storage_validation.json` and is below 20 GiB by a wide margin. Complete trajectories were stored as Zstandard-compressed Parquet, not as a cache substitute.

## Provenance and implementation

The scientific run used repository commit `91017136c3d91120e73af566616736ae817c0dc8`. Final metadata/report code is on branch `eidosoma/groups/42` at commit `1fe61d0cad541efe7d179e486bfdc8b246bde5c2`. Every repository change was committed and pushed before finalization. The first metadata-finalizer call found only a self-reference error—the checker expected `artifact_completeness.json` before writing it—after all scientific/cardinality checks passed. Commit `1fe61d0` corrected that output-order check, and the finalizer was rerun without recomputing any scientific artifact.

Git-backed implementation paths are:

- `src/e01_strict_mrr/core.py`: sampling, preprocessing, partitions, and expanding estimator.
- `src/e01_strict_mrr/analysis.py`: labels, associations, whole-trajectory diagnostics, and claim classification.
- `src/e01_strict_mrr/intervention.py`: complete candidates, null envelope, separation, treatment, and pairing.
- `scripts/e01/run_s12_strict_mrr.py`: bounded orchestration and artifact emission.
- `scripts/e01/finalize_s12_artifacts.py`: outcome-independent metadata correction and final completeness/hash validation.
- `tests/e01/test_s12_strict_mrr_preregistration.py` and `tests/e01/test_s12_strict_mrr.py`: freeze and implementation validation.

The final `artifact_manifest.json` records 53 files excluding itself, for 54 total S12 files, together with every file's size/SHA-256 and an aggregate SHA-256 over the sorted inventory. `trajectory_manifest.json` provides per-trajectory identities/checksums; `seed_manifest.parquet` provides stream identities; `immutable_input_audit.json` preserves pre/post evidence hashes; and `status.json` carries the required machine-readable step handoff.

## Artifact inventory

The final manifest is authoritative. Main artifact groups are:

- **Freeze/provenance:** `preregistration.yaml`, two amendment YAML files, their records, `immutable_input_audit.json`, `seed_manifest.parquet`, `trajectory_manifest.json`, `runtime_manifest.json`, `storage_validation.json`, `artifact_manifest.json`, and `status.json`.
- **Baselines:** `baseline_matrices.npz`, `baseline_observations.parquet`, `baseline_trajectory_events.parquet`, `preprocessing_status.parquet`, `replicator_labels.parquet`, `expanding_estimates.parquet`, `post_fission_estimates.parquet`, and `partition_history.parquet`.
- **Coverage/scientific summaries:** `first_eligibility.csv`, both coverage CSVs, `suppression_summary.csv`, `association_results.csv`, `whole_trajectory_estimates.parquet`, `whole_trajectory_local_values.parquet`, `whole_descriptive_analysis.csv`, and `claim_status_matrix.csv`.
- **Interventions:** feasibility and pilot JSON, complete trajectory/event/estimate/partition/label Parquet files, candidate scores, action log, pairing audit, branch status, Phi summary, and paired result CSV.
- **Validation:** `regeneration_validation.json`, `numerical_validation.json`, `validation_summary.json`, `failure_ledger.csv`, `execution_attempt_001_failure.json`, and `artifact_completeness.json`.
- **Figures:** molecular-step coverage, generation coverage, and selected strict expanding trajectories under `figures/`.

## Caveats, failed assumptions, and interpretation boundary

- Wide late-time coverage does not imply fixed-window validity. A growing prefix and a short local window answer different questions; S11/S11R proved the latter ineligible.
- Eligibility began only after 512 adjacent materialized transitions and a qualifying post-fission partition. Nothing in S12 estimates events before each trajectory's first eligible time.
- The historical GARD branch is a source-traceable public behavior reconstruction. It differs from paper prose on event kernel, fission, initialization, and maximum-step semantics and is not the unavailable author code.
- Historical `H>0.9` labels are retrospective and source-traceable, not author implementation identity. The past-only cosine label is a validation companion, not a paper default.
- The positive association hypothesis was not supported in the valid restricted region. This cannot be reframed as evidence about early pre-replicative spikes.
- Intervention scoring was computationally reproducible but scientifically almost always nonseparable after controlling the full candidate set. A near-universal no-op is a finding about action discriminability, not evidence for equivalent biological outcomes.
- The one applied deletion produced one later divergence and a lower label count for that max trajectory; it is a single, selected, non-generalizable observation.
- Pinned-source nonfinite decompositions and numerical-tolerance misses remain visible. No regularization, tolerance relaxation, hidden default, CCS fallback, or OmegaID substitute was introduced.
- Cross-platform bit identity is not claimed. Exact replay is limited to the recorded Python/NumPy/platform/thread/precision identities; other platforms are bounded only by the recorded tolerance policy.

## Outcome and recommended next action

S12 is **constraining/contradictory**. The strict estimator was widely available after the 512-sample boundary, satisfying the numerical-availability part of the feasibility question. It did not satisfy the coherent-effect criterion: association directions were uniformly negative at the summary level, action discriminability suppressed 99.9083% of authorized treatment opportunities, whole local source evidence was incomplete, and actual-trajectory source/CPU/GPU cross-checks failed the frozen joint tolerance in 14/36 cases.

Accordingly, a restricted S13 scale-up is **not scientifically warranted on this evidence and is not authorized**. E01 should be treated as a partial forensic replication with fixed-window, early-time, early-intervention, Figure 6, and Table 1 claims underdetermined. If alternative reaction-coordinate or causal-architecture work is desired, it should move to a separately preregistered E02 methodological branch after human review. S12 stops here.
