# S10 full results — Validate the information-dynamic implementation

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10 — Validate the information-dynamic implementation** |
| Completion status | **Complete** on 2026-08-01. Only S10 was executed; S11 was not begun. |
| Artifacts written | An outcome-blind preregistration and verification record; Git-backed source wrappers, analytic oracles, synthetic generators, strict sample gate, partition evaluators, builder, and tests; 448 benchmark cases; 8,512 atom results; 2,176 theory comparisons; 368 invariance/shuffle controls; 92 source/backend comparisons; 864 MIB partition results and 1,656 candidate scores; a 124-row branch eligibility table; 232-record seed manifest; ten failure injections; source, runtime, registry-preservation, and artifact manifests; versioned information-dynamics and eligibility contracts; three inspected figures; and this report. |
| Validation result | **CONSTRAINED: 11/12 preregistered gate families passed.** All 448 pinned-reference benchmark cases were eligible; lattice closure was at most `2.22045e-16` nats; 4/4 MATLAB-fixture, 44/44 phyid-versus-OmegaID CPU, and 44/44 OmegaID CPU-versus-GPU comparisons passed; 128/128 time-shuffle controls passed; all three primary MIB objectives recovered the planted split in 8/8 replicates; and 10/10 injected failures were detected. The combined affine/relabel gate failed because OmegaID CPU and GPU each failed 16/16 discrete relabel controls, while phyid passed 64/64 discrete controls and every Gaussian backend passed every affine control. Focused tests passed 10/10 before the plan handoff update; 92/92 non-state-sensitive E01 tests passed, with two intentionally identity-sensitive historical-artifact tests deselected. |
| Outcome classification | **Constraining/contradictory.** The primary success criterion required every applicable backend branch to satisfy its preregistered controls. Pinned phyid and guarded Gaussian OmegaID paths met their bounded gates, but OmegaID's discrete 2×2 path did not preserve independent binary relabelings and is ineligible. No author or paper-primary branch was selected. |
| Caveats or blockers | This validates explicit reconstruction branches, not the unavailable author implementation or MATLAB RNG. The paper-to-atom mapping, redundancy function, estimator, regularization, MIB mapping/objective/normalization/search, and paper-primary choice remain unresolved. Pinned phyid CCS is experimental; OmegaID's more-than-2×2 doublet lattice is not the 16-atom PhiID lattice; equal-width OmegaID vectors cannot represent arbitrary 99-component cuts. Most importantly, the strict validated sample gate requires at least 512 effective samples, so **0/16 queued S11 fixed-window/lag pairs are eligible**. |
| Lay summary | The reference implementation behaved correctly on systems designed to contain no information flow, duplicated information, XOR synergy, and directional autoregressive coupling. Its answers matched exact or analytic calculations, and its Gaussian outputs matched the optional GPU implementation to numerical precision. One accelerated discrete branch failed a basic relabeling test because it thresholds the whole two-series matrix using one global mean; changing numeric labels can therefore change its answer even when the underlying binary states do not. That branch is excluded. The validated estimator also needs more observations than any fixed window currently proposed for S11. |
| Recommended next action | **Stop and return control to the Chief Scientist.** Before separately authorizing S11, either limit it to expanding/whole-trajectory scopes after they reach 512 effective samples or preregister and validate a distinct small-window or regularized estimator branch. Do not relax the sample gate silently, use OmegaID discrete/doublet paths, or choose an author/paper-primary method from these synthetic outcomes. |

## Frozen question and decision

**Frozen question:** Do the pinned phyid reference implementation and explicitly named OmegaID acceleration branches reproduce preregistered theoretical, qualitative, invariance, source-fixture, and backend behavior on synthetic information-dynamic systems, while failing closed on unsupported inputs and preserving every author-method ambiguity?

**Decision:** Partly, with a scientifically meaningful constraint. The pinned phyid Gaussian and binary branches passed their applicable benchmark families. The guarded OmegaID Gaussian 2×2 CPU and GPU paths agreed with phyid and with each other under the frozen binary64 tolerances. The OmegaID discrete 2×2 CPU and GPU paths failed every independent binary-relabel control and are ineligible. Because the preregistered overall rule required every applicable branch to pass, the S10 outcome is constraining/contradictory rather than supportive.

The decision is branch-specific. It does not identify the unavailable author implementation, infer a MATLAB random stream, map the paper's scalar to a source-named atom, or select MMI, CCS, an estimator, a partition representation, an MIB objective, a normalization, or a search as the paper default.

## Lay summary

Information dynamics decomposes predictive information into 16 terms describing redundant, unique, and synergistic contributions from the past to the future. Small numerical or bookkeeping mistakes can still produce plausible-looking values, so S10 tested the implementation on systems whose behavior is known in advance.

Independent noise produced essentially zero lagged information. Two copies of the same Markov bit concentrated information in the redundant atom. An XOR system concentrated information in a synergistic atom. Gaussian systems agreed with population covariance calculations and correctly recovered a directional coupling. Shuffling time destroyed the lagged structure, while harmless affine rescaling left Gaussian answers unchanged. These are strong implementation checks, though not evidence about the paper's biological conclusions.

The optional OmegaID GPU path was extremely close to the CPU calculations for eligible Gaussian inputs. Its discrete path, however, changes how it binarizes data when the numerical labels are transformed: the pinned code compares the complete two-row matrix to one global mean rather than thresholding each series independently. Both CPU and GPU versions consequently failed all relabeling controls by as much as about one nat-scale bit (`0.69315` nats). S10 preserves that failure and excludes this branch.

Finally, the strict estimator gate was validated only for at least 512 effective observations. The S11 queue proposes windows of 32, 64, 128, and 256 observations at lags 1, 2, 4, and 8; every such window has fewer than 512 effective pairs. S10 therefore does not authorize those fixed windows. This is an eligibility constraint, not permission to tune the gate after seeing scientific data.

## Inputs and evidence boundary

### Governing, paper, and upstream inputs

Before implementation or outcome inspection, S10 refreshed and froze 28 inputs:

- `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and the pre-S10 `/workspace/RESEARCH_PLAN.md`;
- `/workspace/input-attachments/MANIFEST.json`, the required `_metadata/ATTACHMENT.md` sidecar, and the supplied paper extraction;
- the official arXiv v1 paper PDF recovered in S03, SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`;
- the S01–S09 canonical full-results reports;
- registry v0.3.0, the frozen source and environment manifests, precision policy, S06 seed contract, S09 compositional-transform contract and specifications, S09 valid-transform registry, and S09 validation/preservation records; and
- the pinned phyid and OmegaID source identities and source-bearing fixture.

All 28 hashes matched the preregistration. Exact paths and hashes are in `preregistration_record.json`. Key identities were:

| Input | Frozen identity or SHA-256 |
| --- | --- |
| S01 report | `4e7025fcb2aaa63eb9fad6b0760e5051b857245cfc4fd4c8840d628e44d72a97` |
| S02 report | `9bcd86ef86c371ba57e991a3bc8295cd92ef2fe05b9edf53327814b0c52f2cfa` |
| S03 report | `7ef7837bf0b2b65fd011ec5b530b90e07089df637050155b81fc08d3c99992ee` |
| S04 report | `62a1f862e17579769aca3939b69eba3ff725078d593ece6b29ccb58a29b8c59d` |
| S05 report | `e83620da619d05687d218186cd2d2789ce26bbac6234de71990948915ea95196` |
| S06 report | `206482bce8e8a47d5050e83c4a99370c2bedf7dd4244a066078f9ad5c3233595` |
| S07 report | `9262b881f1dc15392d1d674e3ac15222dc6dfef69d82ca6aeb74f9fe90fd876d` |
| S08 report | `aed1335f670ca695634aed6023247a6b75c5873a96aa0ff3be8bb5fb22cddbb6` |
| S09 report | `6d78777fd4a4fcecbfc8e0844554b8aa3398c7347464958787d385bfd06aad1c` |
| Registry v0.3.0 | `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891` |
| Source manifest | `50a569e30826fe893258f5c0935469576008f43134eee94e8bf2654d4ef23ed5` |
| S06 seed contract | `a4c5586fc6be012afaff21f47fae422c4d6b6c68200c236df4a5b1ea5e736bb1` |
| S09 transform contract | `7203d0d273f529b3953c45a2c6e1a64dbbd323aac00d72bb60b099ce93dd2679` |
| S09 valid-transform registry | `a09608fe117c23d398a386326a06acc3742e2285aa77d59ff3e3a43590147e56` |

### Source identities and limits

The CPU reference was the pinned public phyid repository at commit `6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44`, tree `fdfe5a21190062b9dda7c8831f72438d8ff5ea95`, under BSD-3-Clause. S10 traced estimator dispatch, mutual-information terms, the 16-atom linear system, Gaussian and binary entropy, and MMI/CCS redundancy to exact files and line ranges recorded in `source_traceability.json`.

The optional accelerated source was OmegaID v0.2.5 at commit `7fcf1fa8e288e0634f81423283d2b349ed88440e`, tree `33ab6f59592048e78a691ecffd9a3dff6d95e54d`, also BSD-3-Clause. Its 2×2 path mirrors the 16-atom lattice. Its more-than-2×2 path instead uses an approximate least-squares doublet lattice with different atom identities and does not use its redundancy argument; S10 never treated that path as a PhiID substitute.

The pinned MATLAB regression fixture was `PhiID-test-simple-1.mat`, SHA-256 `977806cbf913d803c29aceab697edf6c57c8e78188ccd5b13e1fecc4a9b32463`. It contains only six effective samples, so it was used solely for source regression and explicitly bypassed the science sample gate. Passing it is not a claim of author-code or MATLAB-RNG identity.

### Paper equation boundary

The paper prints

\[
\Phi^r = I((X_t,Y_t);(X_{t+1},Y_{t+1}))
- I(X_t;(X_{t+1},Y_{t+1}))
- I(Y_t;(X_{t+1},Y_{t+1})).
\]

Algebra over the pinned 16-atom lattice gives the explicit signed aggregate

\[
\mathrm{str}+\mathrm{stx}+\mathrm{sty}+\mathrm{sts}
-\mathrm{rtr}-\mathrm{rtx}-\mathrm{rty}-\mathrm{rts}.
\]

S10 versioned this as `E01-S10-AGG-PAPER-EQUATION-v1.0.0`. It is an equation-derived aggregate, not a source-named atom and not evidence that the unavailable authors used this exact implementation mapping.

No scientific trajectory, GARD baseline run, or S11 analysis was used in S10. All outcomes come from pinned source fixtures or preregistered synthetic systems.

## Preregistration chronology and outcome firewall

The outcome-blind contract `E01-S10-information-dynamics-preregistration-v1.0.0` froze all inputs, systems, expected behavior, sample sizes, estimators, 16 atoms, aggregate formulas, tolerances, precision, seeds, MIB branches, failure injections, and pass rules. It was committed and pushed at `02d38634f47e73d95882846aeeb89820f38a98b0` before canonical outcomes were inspected. Its SHA-256 is `5c54b8f88e8e8634a4b7f39783e3359084e25ce44cbed2291a141da85e19f3dd`.

Implementation was committed at `6ce76572712aa8e05e774876fb7a4c9f1f6fcb91`. Two post-outcome corrections changed provenance or branch classification, not synthetic data, tolerances, tests, or numeric gates:

1. `847bf9bd805c51c89dd2e8dd929ce68e54a4cc03` retained the observed OmegaID discrete relabel failure as a distinct ineligible branch instead of letting successful Gaussian comparisons obscure it.
2. `28ab6baea2bad5c82f1719bc555daefcc389da89` gave each transformed synthetic dataset a distinct S06 seed/provenance identity. It did not change generated base data or estimator outputs. The final canonical numerical run used this clean commit.
3. `6d6106f970ca721720b8770bc84f6b429cb96f46` reconciled the eligibility report with the already frozen minimum of 512 effective samples and the pre-existing S11 window queue. This was a reporting-only eligibility restriction: it changed no benchmark outcome or gate and made explicit that all queued fixed windows are ineligible.

These changes are disclosed because a provenance fix or a narrower eligibility statement must not be presented as if frozen before outcomes. No observed outcome was used to alter a tolerance, sample size, system, seed, theoretical target, or primary comparison gate.

## Detailed methods

### 1. Versioned atom and estimator mappings

The complete native atom catalog was frozen as:

`rtr, rtx, rty, rts, xtr, xtx, xty, xts, ytr, ytx, yty, yts, str, stx, sty, sts`.

Every computation required all and only these 16 keys. Three diagnostic aggregates were also explicit: total lagged MI as the sum of all atoms, past redundancy as `rtr+rtx+rty+rts`, and past synergy as `str+stx+sty+sts`. An unknown or missing key caused configuration failure.

The estimator catalog retained six executable validation identities and two unresolved, unimplemented candidates:

- strict phyid Gaussian sample-covariance on CPU, using per-series sample-standard-deviation scaling, no bias correction, no regularization, and nats;
- phyid empirical binary plug-in on CPU, using each series' own sample mean, native bits converted explicitly to nats;
- guarded OmegaID 2×2 Gaussian on NumPy CPU and CuPy GPU, permitted only when the strict full-rank gate precluded its source SVD-plus-`1e-6` fallback;
- OmegaID empirical binary 2×2 on CPU/GPU, retained through its relabel failure;
- an independent analytic Gaussian log-determinant oracle for MMI population means; and
- an independent exact-enumerated binary PMF oracle for MMI and CCS.

The kNN and shrinkage candidates remained `UNRESOLVED`/not implemented. MMI and CCS remained separate redundancy branches. MMI was not promoted because it is a library default; CCS remained experimental because the pinned source implements it while its docstrings say “To be implemented” and measure-level tests are expected failures.

### 2. Strict sample and covariance gate

For lag `tau`, effective sample count was `raw_length - tau`. The strict validation branch required at least 512 effective samples and at least 20 samples per joint dimension. Gaussian inputs also required finite values, nonzero sample standard deviations, a full-rank positive-definite four-vector covariance, numerical rank equal to joint dimension, and condition number no larger than `1e12`. Binary inputs required finite values and both thresholded states in each scalar series. No row could be deleted.

Every failure was retained with a machine-readable reason. Exact-copy Gaussian input was expected to be singular and therefore was evaluated only by the discrete estimator; it was not regularized into eligibility. OmegaID's SVD-plus-`1e-6` fallback was likewise prohibited from carrying the strict label.

### 3. Synthetic systems and frozen sample sizes

All primary time series had 32,769 raw observations and 32,768 effective lag-one pairs. Primary systems used 16 replicates; source/backend checks used four designated replicates; the four-dimensional MIB system used eight replicates. The exact frozen systems were:

| System ID | Construction | Preregistered behavior |
| --- | --- | --- |
| `E01-S10-SYS-INDEPENDENT-GAUSSIAN-v1.0.0` | Two independent standard-Gaussian white-noise series | Zero total lagged MI, zero atom means, zero paper aggregate |
| `E01-S10-SYS-REDUNDANT-DISCRETE-v1.0.0` | Two exact copies of a stationary binary Markov state with flip probability 0.1 | Lagged MI `ln(2)+0.9 ln(0.9)+0.1 ln(0.1)`, dominant `rtr`, negative paper aggregate |
| `E01-S10-SYS-REDUNDANT-GAUSSIAN-v1.0.0` | Two noisy observations of an AR(1) latent state, coefficient 0.9 and observation-noise SD 0.35 | Positive total MI and `rtr`, past redundancy above synergy, negative paper aggregate |
| `E01-S10-SYS-XOR-DISCRETE-v1.0.0` | `x_(t+1)=x_t XOR y_t`; future `y` is an independent fair bit | Total/past synergy `ln(2)`, dominant `stx`, zero past redundancy, positive paper aggregate |
| `E01-S10-SYS-COUPLED-AR-v1.0.0` | Gaussian VAR(1), transition `[[0,0],[0.85,0.25]]`, innovation SDs `[1,0.5]`, burn-in 4,096 | Positive total MI, positive forward cross-MI, population-zero reverse cross-MI, correct direction |
| `E01-S10-SYS-BLOCK-AR4-v1.0.0` | Four observed series from two independent AR blocks, coefficient 0.82 and observation-noise SD 0.20 | MIB procedures recover `[0,1] | [2,3]` |

### 4. Theoretical and qualitative comparisons

The exact binary PMFs and Gaussian stationary covariances were constructed independently of the source wrappers. For Gaussian MMI, population entropies and mutual informations used analytic log determinants and the same explicit lattice equations. For binary MMI/CCS, all supported states were enumerated and converted from source-native bits to reported nats.

The preregistered tolerances were intentionally wider than binary64 backend tolerances because they cover finite-sample estimation:

- independent Gaussian ensemble total MI and maximum atom magnitude: `0.002` nats; every replicate total MI: `0.01`;
- exact binary theory: ensemble maximum atom error `0.01`, replicate maximum `0.04`, at least 15/16 passing;
- Gaussian MMI population theory: ensemble maximum atom error `0.015`, replicate maximum `0.05`, at least 15/16 passing; and
- qualitative direction/dominance gates: at least 15/16 passing.

### 5. Transform and shuffle controls

Gaussian affine invariance applied `x'=17x+23` and `y'=-0.125y+5`. Discrete relabeling applied `x'=3x+11` and `y'=-2y+7`. Every eligible estimator/redundancy branch had to preserve every mean atom and intermediate MI within absolute and relative tolerance `1e-10`.

The time-shuffle control applied one separately seeded common permutation to complete contemporaneous rows before reconstructing lag-one pairs. This preserves same-time marginals and cross-sectional structure while destroying serial order. Shuffled absolute total MI had to be below `0.01` nats and below 10% of the original, with at least 15/16 replicates passing.

### 6. Source and CPU/GPU cross-checks

The MATLAB fixture was compared for Gaussian MMI, Gaussian CCS, discrete MMI, and discrete CCS at local/mean absolute tolerance `1e-8` and relative tolerance `1e-5`. Its science-sample status remained ineligible.

For supported 2×2 cases, phyid and OmegaID NumPy CPU were compared locally and by atom means at `1e-10` absolute and relative tolerance. OmegaID NumPy CPU and CuPy GPU were compared at the same tolerance. CuPy used binary64 on one recorded NVIDIA L4. TF32 and mixed precision were disabled.

### 7. Explicit MIB branches

The four-dimensional planted system evaluated three mappings, three objectives, three normalizations, and four executable searches without selecting a paper default.

Mappings were group arithmetic means, per-part PC1 with deterministic sign/eigentolerance rules, and OmegaID equal-width vector parts. Objectives were synchronous Gaussian MI, bidirectional lagged Gaussian MI, and the absolute paper-equation aggregate. Normalizations were none, minimum-part entropy, and geometric part size. Searches were exhaustive all unordered nonempty bipartitions, exhaustive balanced cuts, a fixed deterministic spectral candidate, and deterministic greedy single-component flips.

Unresolved author mapping/objective/normalization/search sentinels remained non-executable. Queyranne search was not implemented because submodularity was not established for these candidate objectives. Every search emitted the selected canonical partition, objective, normalization denominator, mapping diagnostics, comparison to exhaustive search, status, and reason. The primary preregistered branch was group mean, no normalization, exhaustive all, separately for all three objectives.

### 8. Randomness and provenance

All stochastic inputs used S06's domain-separated SHA-256-to-PCG64DXSM contract with root seed `1010…1010` (64 hexadecimal digits) and the `estimator` stream purpose. Every system, replicate, transform, shuffle, and MIB dataset had a complete independent seed identity. The final manifest contains 232 records with 232 unique stream IDs, seed materials, seed-payload hashes, and data hashes. No identity is claimed to match MATLAB or author code.

### 9. Failure injection

Ten preregistered faults were tested: unresolved estimator sentinel; unknown/missing atom; singular exact-copy Gaussian covariance; hidden nonfinite-row deletion; bits mislabeled as nats; CPU/GPU perturbation above tolerance; noncanonical partition/tie; shuffle without permutation identity; OmegaID fallback mislabeled strict; and checksum tampering. All ten were detected and preserved in `failure_injection.json`.

## Parameters, dependencies, and compute

| Item | Frozen or observed value |
| --- | --- |
| Reference | phyid commit `6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44` |
| Accelerator | OmegaID v0.2.5 commit `7fcf1fa8e288e0634f81423283d2b349ed88440e` |
| Python / NumPy / SciPy | 3.13.14 / 2.4.6 / 1.18.0 |
| PyYAML | 6.0.3 |
| CuPy / fastrlock | 13.6.0 / 0.8.3 |
| GPU | NVIDIA L4, UUID `GPU-1f5bed19-d63f-be98-470c-f670e3d4aafd`, driver 610.43.02, CUDA runtime 12.9 as reported by CuPy |
| Precision | CPU/GPU `float64`; TF32 off; mixed precision off |
| Backend tolerance | Absolute `1e-10`, relative `1e-10` |
| Workers | Four process workers; OMP/OpenBLAS/MKL/NumExpr each restricted to one thread |
| Canonical runtime | 176.269 seconds at clean repository commit `28ab6baea2bad5c82f1719bc555daefcc389da89` |
| New cached dependency wheels | `cupy-cuda12x==13.6.0`, SHA-256 `52d9e7f83d920da7d81ec2e791c2c2c747fdaa1d7b811971b34865ce6371e98a`; `fastrlock==0.8.3`, SHA-256 `dbdea6deeccea1917c6017d353987231c4e46c93d5338ca3e66d6cd88fbce259` |

The isolated environment `/cache/e01_s10/venv` was created with Python 3.13 system-site packages. Only the pinned CuPy and fastrlock wheels were added, from `/cache/e01_s10/wheelhouse`; no project or system dependency file was changed. Four worker processes stayed within the eight-core workspace limit and numerical-library threading was disabled to avoid nested parallelism.

## Results

### Gate-family summary

| Preregistered family | Result | Anchor evidence |
| --- | --- | --- |
| Lattice closure | PASS | Maximum absolute error `2.22045e-16` nats versus `1e-10` |
| Independent Gaussian | PASS | Ensemble total MI `4.87173e-05` nats; maximum gate atom magnitude `8.68446e-06` |
| Exact discrete theory | PASS | 64/64 replicate atom vectors passed; every system/redundancy had at least 15/16 passing |
| Gaussian MMI population theory | PASS | 16/16 ensemble atom comparisons passed; worst ensemble atom error `0.00521241` nats |
| Qualitative systems | PASS | Redundant Gaussian and coupled AR: 16/16 per redundancy family |
| Affine and relabel invariance | **FAIL** | 32/240 controls failed, all and only OmegaID discrete CPU/GPU relabel cases |
| Time shuffle | PASS | 128/128; worst absolute shuffled MI `0.000280055` nats and worst original ratio `0.000404031` |
| Source MATLAB fixture | PASS | 4/4; maximum local absolute error `5.32907e-15` |
| Reference versus OmegaID CPU | PASS | 44/44; maximum local absolute error `3.10862e-13` |
| OmegaID CPU versus GPU | PASS | 44/44; maximum local absolute error `2.11386e-13` |
| MIB recovery | PASS | Each of three primary objectives recovered the planted cut 8/8 |
| Failure injection | PASS | 10/10 detected |

The machine-readable field `referenceGateSuccess` is false because the preregistered aggregate named `affineAndScaleInvariance` includes the optional OmegaID discrete controls as well as phyid controls. It must not be read as a claim that phyid failed: phyid passed all 160 of its applicable affine/relabel comparisons. The overall false result correctly enforces the preregistered rule that a failed optional branch remains visible and ineligible.

### Synthetic-system anchors

The following are ensemble means over 16 base replicates. Total MI and the paper-equation aggregate are the same under MMI and CCS for these systems even where individual atoms differ.

| System | Mean total MI (nats) | Mean paper aggregate (nats) | Key comparison |
| --- | ---: | ---: | --- |
| Independent Gaussian | `0.00004872` | `0.0000000154` | Consistent with zero |
| Redundant binary copies | `0.36748318` | `-0.36748318` | Theory `0.36806421`; dominant `rtr` |
| Redundant Gaussian copies | `0.78039877` | `-0.73816049` | Population total `0.78383933`; population aggregate `-0.74157950` |
| XOR discrete | `0.69314181` | `0.69307438` | Theory `ln(2)=0.69314718`; dominant `stx` |
| Coupled Gaussian AR | `0.71191307` | `0.08250202` | Population total `0.71147384`; forward cross-MI positive, reverse population cross-MI zero |

Across all 2,176 recorded theoretical comparisons, the largest ensemble atom error was `0.00521241` nats for redundant-Gaussian MMI `rtr`, below `0.015`. Exact discrete maximum ensemble atom errors were `0.00058102` for redundant copies, `0.0000799` for XOR MMI, and `0.00203162` for XOR CCS, all below `0.01`. Coupled-AR MMI's maximum ensemble atom error was `0.00115879` nats.

### Invariance and the constraining OmegaID result

| Backend and control | Passes / total | Maximum absolute error (nats) | Eligibility consequence |
| --- | ---: | ---: | --- |
| phyid Gaussian affine | 96/96 | `5.77021e-15` | Conditionally eligible subject to sample/preprocessing gates |
| phyid discrete relabel | 64/64 | `0` | Validated for synthetic binary cases, not continuous S11 input |
| OmegaID CPU Gaussian affine | 24/24 | `4.87544e-15` | Conditional accelerator at effective `n>=512` |
| OmegaID GPU Gaussian affine | 24/24 | `4.66294e-15` | Conditional accelerator at effective `n>=512` |
| OmegaID CPU discrete relabel | **0/16** | `0.69314955` | Ineligible |
| OmegaID GPU discrete relabel | **0/16** | `0.69314955` | Ineligible |

Source inspection explains the discrete failure: pinned phyid thresholds each scalar series by its own mean, whereas pinned OmegaID's `_binarize(v)` compares the complete matrix against one global mean. Independent monotone/affine relabeling of one series can therefore change the jointly thresholded state assignment. CPU and GPU agree with each other on this behavior, which confirms backend reproduction of a source behavior but does not make the behavior scientifically invariant.

### MIB and partition diagnostics

The primary group-mean/no-normalization/exhaustive-all branches recovered `[0,1] | [2,3]` in 8/8 replicates for each objective. Minimum winner margins across replicates were `0.0972382` for synchronous MI, `0.1213681` for bidirectional lagged MI, and `0.000515043` for the absolute paper aggregate.

Of 864 partition-result rows, 720 were eligible and 144 were explicitly ineligible. The 1,656 candidate-score rows preserve all evaluated cuts. Against exhaustive search:

- group-mean greedy and spectral matched in 72/72 eligible comparisons each;
- Omega equal-width vector spectral matched in 72/72, while exhaustive-all and greedy were ineligible for unbalanced search domains (144 retained ineligible results);
- PC1 greedy matched in 72/72; and
- PC1 spectral matched in 69/72, with the three absolute-paper-equation cases retained as approximation-gate failures.

The eligibility table has 124 records: 87 validation-only eligible partition records, 21 explicitly ineligible partition records, eight unresolved non-executable sentinels, two conditionally eligible phyid Gaussian redundancy branches, two conditionally eligible OmegaID Gaussian accelerators, two ineligible OmegaID discrete accelerators, one binary-only phyid estimator, and one ineligible OmegaID doublet substitute. These are mutually exclusive status counts; the exact branch records are in `eligibility_registry.csv`.

None of these MIB branches is an author default. Exhaustive evidence is only four-dimensional, Omega vector mapping requires equal widths, and the approximation results do not establish scalability to the paper's 99-component system.

### S11 branch and window eligibility

The versioned eligibility registry now separates numerical branch validity from analysis-scope validity:

| Candidate | S11 status | Boundary |
| --- | --- | --- |
| phyid Gaussian strict, MMI | Conditionally eligible | Effective sample count at least 512; named preprocessing and complete MIB specification required |
| phyid Gaussian strict, CCS | Conditionally eligible | Same sample gate; CCS remains experimental and must remain a distinct sensitivity branch |
| OmegaID Gaussian 2×2 CPU | Conditional accelerator | Only guarded full-rank 2×2 path at effective `n>=512`; no SVD-plus-`1e-6` fallback |
| OmegaID Gaussian 2×2 GPU | Conditional accelerator | Same, plus recorded float64 GPU precision identity |
| phyid binary | Synthetic binary validation only | Not eligible for continuous compositional S11 input |
| OmegaID binary CPU/GPU | Ineligible | Relabel-invariance failure |
| OmegaID more-than-2×2 doublet lattice | Ineligible | Different approximate lattice, not a 16-atom substitute |
| kNN, shrinkage, author/source-recovered choices | Unresolved/non-executable | Not implemented or validated |

For every queued fixed S11 pair, effective observations equal `window length - lag`:

| Window | Eligible lags from `{1,2,4,8}` under minimum 512 | Status |
| ---: | --- | --- |
| 32 | None | Ineligible |
| 64 | None | Ineligible |
| 128 | None | Ineligible |
| 256 | None | Ineligible |

Thus 0/16 queued fixed-window/lag pairs are eligible. Expanding or whole-trajectory analyses become conditionally eligible only after reaching 512 effective pairs and passing rank, conditioning, finite-value, and preprocessing checks. A small-window or regularized estimator is not a harmless parameter change; it requires a new outcome-blind preregistration and synthetic validation before scientific use.

## Validation and quality checks

### Automated checks

- The preregistration verified 28/28 frozen input hashes and was committed before outcome inspection.
- All 448 benchmark cases were retained and eligible: 160 base, 160 affine/relabel, and 128 time-shuffle cases. Maximum observed eligible covariance condition number was `172.374`, far below `1e12`.
- The atom table contains 8,512 rows, exactly 16 atoms per case plus the separately recorded fixture/backend atoms represented by its contract.
- All 16 atom sums and both aggregate closures passed; maximum lattice closure error was `2.22045e-16`.
- All backend and invariance failures were status-bearing; no row was removed.
- All 232 seed records had distinct stream ID, seed material, seed-payload hash, and dataset hash.
- Registry v0.3.0 was byte-for-byte unchanged at SHA-256 `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`, remained non-executable, and retained `noSilentDefaults: true`.
- The three figures were visually inspected at 2340×1260, 1620×900, and 2160×900 pixels and were legible and consistent with their source tables.
- S11 output was absent at handoff.

### Repository tests and lint

Before the plan was advanced to its S10-complete handoff state, the focused S10 suite passed 10/10. The complete non-state-sensitive E01 suite passed 92/92 with two identity-sensitive historical-artifact tests deliberately deselected: the S09 builder test freezes the pre-S09 `RESEARCH_PLAN.md` hash, and the S06 regeneration test freezes the earlier repository capture identity. Advancing those upstream identities is expected; their historical artifacts were not mutated.

After the final reporting-only sample-eligibility restriction, nine non-preregistration focused tests passed and Ruff formatting/checks passed. The preregistration hash test was not rerun against the advanced plan because it correctly expects the pre-S10 plan hash preserved in the frozen record. The pinned binary source emits a NumPy divide-by-zero warning when computing `log2(0)` for unobserved states; all observed outputs remained finite, exact fixture parity passed, and the warning is retained rather than suppressed.

## Commands

Principal commands were run from `/workspace/arrival-of-self-replicators`:

```bash
# Refresh, evidence, and identity audit.
sed -n '1,999p' /workspace/AGENTS.md
sed -n '1,999p' /workspace/FULL_PLAN.md
sed -n '1,999p' /workspace/RESEARCH_PLAN.md
jq . /workspace/input-attachments/MANIFEST.json
sha256sum <frozen inputs and pinned source files>
git -C /cache/e01_s03/sources/phyid rev-parse HEAD^{tree}
git -C /cache/e01_s03/sources/omegaid rev-parse HEAD^{tree}

# Freeze before benchmark outcomes.
git add configs/e01/s10_information_dynamics_preregistration.yaml
git commit -m "Preregister S10 information dynamics validation"
git push origin eidosoma/groups/42

# Cached Python 3.13 GPU environment; wheels are pinned in the preregistration.
python -m venv --system-site-packages /cache/e01_s10/venv
/cache/e01_s10/venv/bin/python -m pip install --no-index \
  /cache/e01_s10/wheelhouse/fastrlock-0.8.3-*.whl \
  /cache/e01_s10/wheelhouse/cupy_cuda12x-13.6.0-*.whl

# Formatting and focused validation.
ruff format --check scripts/e01/run_s10_information_dynamics_validation.py \
  src/e01_information_dynamics tests/e01/test_information_dynamics.py
ruff check scripts/e01/run_s10_information_dynamics_validation.py \
  src/e01_information_dynamics tests/e01/test_information_dynamics.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=src pytest -q \
  tests/e01/test_information_dynamics.py

# Broader E01 regression suite, excluding two intentionally historical identities.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=src pytest -q tests/e01 \
  --deselect tests/e01/test_compositional_preprocessing.py::test_builder_writes_complete_lossless_status_bearing_artifacts \
  --deselect tests/e01/test_rng_schema.py::test_generated_artifacts_and_fresh_process_regeneration_when_present

# Canonical numerical run.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
PYTHONDONTWRITEBYTECODE=1 /cache/e01_s10/venv/bin/python \
  scripts/e01/run_s10_information_dynamics_validation.py \
  --artifacts-root /artifacts

# Final report-only consistency and manifest refresh.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=src pytest -q \
  tests/e01/test_information_dynamics.py \
  -k 'not preregistration_is_frozen_and_complete'
/cache/e01_s10/venv/bin/python \
  scripts/e01/run_s10_information_dynamics_validation.py \
  --artifacts-root /artifacts --finalize-manifest-only
git push origin eidosoma/groups/42
```

The wheel-install command above records the reproducible installation form for the two cached pinned wheels; wheel paths and hashes, not a mutable index resolution, are the dependency authority.

## Artifacts written

### Canonical S10 directory

| Artifact | Contents |
| --- | --- |
| `preregistration.yaml`, `preregistration_record.json` | Frozen settings and 28 verified input identities |
| `benchmark_cases.csv` | 448 status-bearing source-reference cases |
| `atom_results.csv` | 8,512 native atom estimates, including source fixture records |
| `theoretical_comparisons.csv` | 2,176 replicate/ensemble oracle comparisons |
| `invariance_results.csv` | 368 affine, discrete-relabel, and shuffle controls |
| `backend_comparisons.csv` | 92 source/CPU/GPU comparisons |
| `mib_partition_results.csv` | 864 selected-partition records |
| `mib_candidate_scores.csv` | 1,656 evaluated candidate cuts |
| `eligibility_registry.csv` | 124 candidate branch/status records, including failures and unresolved sentinels |
| `validation_summary.json` | Gate-family summary and outcome classification |
| `failure_injection.json` | Ten detected fault records |
| `seed_manifest.json` | 232 complete S06 estimator-stream identities and data hashes |
| `runtime_manifest.json` | Python, NumPy, source commit, CPU-thread, GPU/CuPy, repository, and wall-time identity |
| `source_traceability.json` | Pinned source files, hashes, line-level behavior references, and source boundaries |
| `registry_preservation.json` | Byte-for-byte registry preservation and unresolved relevant parameters |
| `synthetic_atom_profiles.png` | Synthetic theory/estimate atom profiles |
| `backend_agreement.png` | Reference/CPU/GPU agreement and failed discrete invariance diagnostics |
| `mib_recovery.png` | Planted-cut recovery and approximation diagnostics |
| `artifact_manifest.json` | SHA-256, size, role, repository, and bundle provenance for all final artifacts |
| `research_step_full_results.md` | This canonical handoff |

### Reusable bundle contracts

- `/artifacts/E01_forensic_replication_bundle/information_dynamics/information_dynamics_contract_v1.0.0.yaml`
- `/artifacts/E01_forensic_replication_bundle/information_dynamics/information_dynamics_eligibility_registry_v1.0.0.yaml`

Repository code remains in Git rather than being duplicated under artifacts. No `status.json` was required by `RESEARCH_PLAN.md` or the workflow for S10.

## Provenance

| Item | Identity |
| --- | --- |
| Repository | `/workspace/arrival-of-self-replicators` |
| Branch | `eidosoma/groups/42` |
| Preregistration commit | `02d38634f47e73d95882846aeeb89820f38a98b0` |
| Implementation commit | `6ce76572712aa8e05e774876fb7a4c9f1f6fcb91` |
| Discrete-branch preservation commit | `847bf9bd805c51c89dd2e8dd929ce68e54a4cc03` |
| Distinct transform-seed identity commit / canonical numeric run head | `28ab6baea2bad5c82f1719bc555daefcc389da89` |
| Final sample-eligibility reporting commit | `6d6106f970ca721720b8770bc84f6b429cb96f46` |
| Remote branch | Pushed through `6d6106f970ca721720b8770bc84f6b429cb96f46` |
| Preregistration SHA-256 | `5c54b8f88e8e8634a4b7f39783e3359084e25ce44cbed2291a141da85e19f3dd` |
| Validation summary SHA-256 | `e86eaf9825c69a2a3bfd27486ce954af4ae7793877b570927ff13d4707a062d9` |
| Seed manifest SHA-256 | `cc546552584cba67d71740207c068c8c66aad0a8da4bb39e5198ddd3eaa41ea1` |
| Information-dynamics contract SHA-256 | `c31adef6d6fc7acf098244422a6a46c7c98b582bbd1d99d431216577c1f01a83` |
| Registry v0.3.0 before/after | `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891` / identical |

The final eligibility-registry and artifact-manifest hashes are recorded by the refreshed `artifact_manifest.json`, avoiding a self-referential report checksum claim here.

## Caveats, blockers, failed assumptions, and limitations

1. **No author-method recovery.** Neither the paper nor pinned repositories identify the unavailable author implementation, atom/aggregate mapping, redundancy function, estimator family, regularization, MIB mapping/objective/normalization/search, evaluation cadence, or MATLAB RNG. Source-library defaults are not author evidence.
2. **OmegaID discrete is ineligible.** CPU and GPU faithfully agree with each other but fail all independent relabel controls because of global matrix-mean binarization. Backend agreement cannot rescue a failed scientific invariant.
3. **CCS is experimental.** The source path runs and passed the frozen synthetic gates, but its own docstrings and expected-failure tests prevent promotion to a primary method.
4. **OmegaID scope is narrow.** The 2×2 Gaussian path is conditionally useful; the more-than-2×2 doublet approximation has a different lattice. Equal-width vector inputs cannot encode arbitrary 99-component bipartitions.
5. **MIB evidence is small-dimensional.** Exhaustive search was validated only at dimension four. Greedy/spectral agreement on this fixture is not evidence of global optimality or paper-scale tractability. PC1 spectral failed 3/72 comparisons.
6. **Preprocessing remains branch-specific.** S09 accepted numerical transform specifications but did not recover an author default. Full CLR and raw closed proportions remain structurally covariance-singular. Every S11 record must retain the complete preprocessing identity.
7. **Fixed-window S11 is blocked.** The frozen strict branch requires at least 512 effective pairs. Every queued window/lag pair has fewer. Relaxing, regularizing, pooling, or deleting rows would create a new unvalidated branch.
8. **Finite-sample success is bounded.** The synthetic systems and tolerances support implementation validation, not proof for all distributions, dimensions, lags, or biological trajectories.
9. **Source fixture is tiny.** Its four comparisons are useful regression evidence but are science-gate-ineligible at six effective samples.
10. **Floating-point scope is explicit.** CPU/GPU agreement is demonstrated for the recorded Python/NumPy/CuPy/L4 binary64 environment within `1e-10`; exact equality across platforms is not claimed.
11. **Warnings are preserved.** Pinned phyid binary entropy evaluates `log2(0)` for unobserved states and emits a runtime warning. Outputs used by the tests were finite and source parity passed, but the warning remains part of the source behavior.
12. **Outcome classification is intentionally strict.** Eleven successful families do not erase one failed branch. The failure narrows eligibility and makes the result constraining/contradictory under the preregistered rule.

## Final handoff

S10 is complete, `RESEARCH_PLAN.md` has been updated to show no active step, and S11 remains unstarted. The validated strict phyid Gaussian MMI/CCS branches and guarded OmegaID Gaussian accelerators are conditionally eligible only for explicitly versioned analysis units with at least 512 effective samples. The queued S11 fixed-window grid has no eligible pair. OmegaID discrete CPU/GPU, the doublet-lattice substitute, unimplemented estimator candidates, and unresolved author choices remain ineligible or non-executable.

Recommended next action: return control to the Chief Scientist. If S11 is later authorized, first choose either an expanding/whole-trajectory design that reaches 512 effective samples or a separate preregistered small-window validation step. Do not begin S11 from this handoff automatically.
