# S06 Full Results — Freeze Random-Number Semantics and the Data Schema

## Top summary

- **Research step ID:** S06
- **Completion status:** Complete; S06 only. S07 was not begun.
- **Artifacts written:** Versioned seed-derivation, trajectory-precision, seed-schema, and trajectory-schema contracts; seed, event, fission, full-trajectory, and regeneration examples; branch-coverage, schema, serialization, seed, precision, regeneration, registry-preservation, validation-summary, and provenance-manifest artifacts; Git-backed implementation, build/replay scripts, and tests.
- **Validation result:** PASS. All 9 canonical stream purposes were unique and matched frozen known-answer vectors; the example and 3/3 explicit S05 branch profiles conformed to Draft 2020-12 schemas and custom invariants; 489 binary64 fields round-tripped losslessly; checksum tampering was rejected; auxiliary streams did not perturb the simulated trajectory; one-thread OpenBLAS execution was observed; and in-process plus fresh-process same-engine regeneration reproduced the payload, checksum, and canonical bytes exactly. Focused tests passed 14/14, and the repository suite passed 49/49.
- **Outcome classification:** Supportive. The bounded S06 hypothesis was met: explicit stream separation prevents the tested accidental couplings and an identical manifest regenerates an identical trajectory under the frozen same-engine identity.
- **Caveats or blockers:** This is a reconstruction-engineering contract, not evidence for the unavailable author implementation or legacy MATLAB RNG behavior. Public historical GARD remains draw-tape-only for exact fixtures and is explicitly nonconforming to canonical seeded replay. Full cross-platform trajectory equality is not promised. Registry v0.3.0 remains non-executable with 64 unresolved items and 21 unexpanded branch sets; `UNRESOLVED::E01-A020`, `UNRESOLVED::E01-A025`, author-code unavailability, and legacy MATLAB uncertainty remain unchanged.
- **Lay summary:** The reconstruction now gives every source of randomness its own named, reproducible random stream and records every state transition needed to audit a run. Replaying the same small run with the same code and environment produced exactly the same file, byte for byte. This does not reveal how the paper authors seeded MATLAB, but it prevents later reconstruction work from quietly inventing or changing that choice.
- **Recommended next action:** Return control to the Chief Scientist. S07 is eligible but must not start without separate authorization.

## Frozen question

**Question:** Do separate seed streams prevent accidental coupling and permit exact regeneration?

**Answer:** Yes, within the explicitly frozen S06 scope. Nine domain-separated streams, immutable seed requests, explicit coupling namespaces, full specification/trajectory identities, canonical lossless serialization, and fail-closed runtime identity checks produced exact same-engine regeneration. The result is not a cross-engine, cross-RNG, cross-platform, or author-code identity claim.

## Lay summary

A stochastic simulation can change when an apparently unrelated random draw is inserted—for example, when a later analysis consumes a number from the same generator used for reaction events. S06 prevents that class of hidden coupling by assigning separate streams to catalytic-matrix generation, initialization, event choice, waiting time, fission, daughter selection, interventions, estimators, and machine learning. Each stream has a versioned identity derived from an explicit root seed and namespace.

The trajectory format records the complete selected specification, seed identities, integer state before and after every event, all event propensities, total propensity, time, generation and stopping information, fission products, and daughter choice. Floating-point values are written as exact hexadecimal binary64 strings rather than rounded decimal JSON numbers. Each payload is protected by SHA-256.

The canonical example contains 3 generations, 14 events, and 3 fissions. Rebuilding it in the same pinned independent engine produced the same payload and serialized file exactly. All three explicit S05 fixture families—historical-comparison categorical events, modern direct Gillespie events, and paper-prose vector-Poisson batches—also passed schema, invariant, and exact-replay checks. These are engineering fixtures, not scientific validation of their stochastic distributions; that remains S07 work.

## Inputs and evidence boundary

The following required sources were refreshed before implementation. Exact input paths, sizes, and SHA-256 values are recorded in `artifact_manifest.json`.

- Workspace instructions and plans: `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and `/workspace/RESEARCH_PLAN.md`.
- Attachment inventory and sidecar, read before the paper payload: `/workspace/input-attachments/MANIFEST.json` and `/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md`.
- Original paper evidence: the supplied Markdown extraction and the verified 18-page arXiv v1 PDF at `/cache/e01_s03/downloads/paper-2607.28250v1.pdf` (SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`).
- S01–S05 canonical full-results reports and their artifact/provenance manifests.
- Registry: `/artifacts/E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml`, version `E01-specification-registry-v0.3.0`, SHA-256 `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`.
- Frozen provenance: `source_manifest.yaml`, `environment_report.json`, and `precision_policy.yaml` under `/artifacts/E01_forensic_replication_bundle/provenance/`.
- Both engine contracts: the S04 historical behavior contract (SHA-256 `e6fe49aba2240047d018e5b619ef07d3e48922fb43a963256b6b2233f07d0a43`) and S05 independent-engine contract (SHA-256 `a35e313cb0685218691397980d1f5d8020fee8c994359e3227b9b1c1ef8605e8`).

The registry was treated as closed evidence. No registry value, conflict, branch set, or sentinel was updated. S06 specifications are explicitly labeled fixtures and cannot be used as recovered author defaults.

## Methods

### Canonical seed identities

The seed contract is `E01-seed-derivation-contract-v1.0.0`, with seed schema `E01-seed-schema-v1.0.0` and derivation algorithm `E01-SHA256-DOMAIN-SEPARATION-PCG64DXSM-v1`.

Each request must explicitly supply:

- experiment, specification, trajectory, replicate, and engine identities;
- a 256-bit root seed encoded as exactly 64 lowercase hexadecimal characters;
- a coupling policy and coupling reason;
- one namespace for every canonical stream purpose.

The nine purposes are:

1. `catalytic_matrix`
2. `initial_state`
3. `event`
4. `waiting_time`
5. `fission`
6. `daughter_selection`
7. `intervention`
8. `estimator`
9. `machine_learning`

The user-required purposes are all present. `waiting_time` is an additional, explicit ninth purpose because the S05 engine contract already separates direct-Gillespie waiting draws from event selection. Folding it back into `event` would have silently regressed that validated separation.

For purpose (p), the canonical context contains the seed-schema version, derivation-algorithm version, experiment ID, replicate index, stream purpose, and explicit namespace. The preimage is:

```text
"EIDOSOMA-E01-SEED-DERIVATION-v1\0"
|| uint32_be(length(root_seed_bytes))
|| root_seed_bytes
|| uint64_be(length(canonical_context_utf8))
|| canonical_context_utf8
```

SHA-256 of that preimage is used in full as an unsigned big-endian 256-bit seed for `numpy.random.PCG64DXSM`; no digest truncation occurs. Stream IDs include the schema version, purpose, and digest. The fixture freezes both derived seed material and the first four raw `uint64` values for each purpose.

`trajectory_isolated` requires every purpose to name the exact canonical trajectory namespace and requires a null coupling reason. `explicit_common_random_numbers` requires every namespace to remain explicit and requires a nonempty scientific reason. Matching namespaces are the only supported way to share a raw stream across trajectories or specifications. Seed-request and seed-bundle mappings are copied into immutable mapping proxies after validation.

### Engine boundary

The S05 independent engine receives six separate live generators: matrix, initialization, event, waiting-time, fission, and daughter-selection. Intervention, estimator, and machine-learning generators are derived and serialized but are not consumed by the simulation. A validation run consumed 101, 103, and 107 raw values from those auxiliary streams before simulation and obtained an identical matrix, initial state, and lineage.

The S04 historical engine is not relabeled as conforming. Its public behavior depends on a shared, order-sensitive legacy MATLAB global RNG. Substituting nine PCG64DXSM generators would change that historical behavior. Exact S04 fixtures therefore remain explicit draw tapes with no RNG-identity inference. The contract records `NONCONFORMING_TO_CANONICAL_SEEDED_REGENERATION` and preserves `UNRESOLVED::LEGACY_MATLAB_RNG_ALGORITHM_AND_GLOBAL_STATE_ORDER`.

### Lossless trajectory schema

The trajectory schema is `E01-trajectory-schema-v1.0.0`. Each envelope contains a canonical serialization version, a SHA-256 of the canonical payload bytes, and the payload. Unknown fields are rejected at the defined object boundaries.

The payload records:

- experiment, specification, trajectory, replicate, engine, repository, source, runtime, schema, precision, byte-order, and numeric-thread identities;
- the complete branch-explicit S05 specification and its field checksum;
- the complete seed manifest and its checksum;
- registry version/hash, execution gate, unresolved counts, branch-set counts, and preserved author/MATLAB/sampling sentinels;
- the exact catalytic matrix and initial state plus RNG state hashes before and after their draws;
- for every generation, its identity, growth initial/final states, event sequence, elapsed time, stopping reason, fission, selected daughter, and post-generation state;
- for every event, integer pre/post states and masses, event kind/index/species, update kernel, boost/join/leave/concatenated/normalized propensities, total propensity, attempted/applied update counts, boundary action, clock values, and event/waiting RNG state hashes;
- for every fission, the integer parent, both children, discarded molecules, conservation result, selection semantics, selected label/state, post-fission semantics, and fission/daughter RNG state hashes;
- terminal state, requested generations, completed fissions, stopping reason, and initial/terminal state hashes for all nine RNG streams.

All stored molecule counts are nonnegative signed-int64 JSON integers. All finite binary64 values use canonical Python `float.hex` strings. Decimal JSON floats, NaN, Infinity, duplicate object keys, noncanonical float strings, checksum mismatch, and noncanonical serialized bytes are rejected. This preserves the exact binary64 bits without depending on decimal parser behavior.

The invariant validator additionally checks specification and identity checksums; sequential generation, event, and RNG state chains; propensity dimensions, nonnegativity, concatenation, totals, and normalization; categorical event identities and update reconstruction; fixed-exposure and Gillespie clocks; mass changes; fission conservation and fixed-size masses; daughter consistency; stopping reasons; terminal state/count consistency; and nonconsumption of auxiliary streams.

### Same-engine regeneration identity

Exact replay is intentionally narrow. It requires exact agreement on:

- repository commit and aggregate independent-engine/adapter source hashes;
- Python and NumPy versions;
- platform, byte order, and the frozen S03 replacement runtime fingerprint;
- seed-schema, trajectory-schema, and precision-contract hashes;
- explicit `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` identities.

The builder also inspected the loaded numeric thread pools and observed OpenBLAS `0.3.31.188.0`, pthreads, SkylakeX, with exactly one thread. Missing or different numeric-thread environment settings reject capture or replay.

### Cross-platform precision boundary

Serialized binary64 values and checksums are portable after generation because their exact bit patterns are stored. The SHA-256 derivation and PCG64DXSM raw-integer known-answer vectors are exact for the frozen algorithm and NumPy version.

Full trajectory regeneration across arbitrary platforms is **not guaranteed**. Distribution-sampler implementation, `exp`, reductions, libm, compiler, and architecture can alter low floating-point bits, and a small propensity change can cross a categorical decision boundary.

For a future cross-platform audit, corresponding finite floating values must satisfy all three limits: absolute error at most (10^{-12}), relative error at most (10^{-12}), and ULP distance at most 8. These bounds never excuse a changed event, integer state, fission, daughter, stopping reason, or checksum. Any such discrete divergence is `CROSS_PLATFORM_REGENERATION_FAILURE`, not approximate success. S06 measured same-runtime replay only: 489 compared float fields were bit-exact and the maximum ULP distance was 0.

## Results

### Seed and coupling results

| Check | Result |
| --- | --- |
| Required stream purposes | 9/9 present |
| Unique stream IDs | 9/9 |
| Unique 256-bit seed material | 9/9 |
| Frozen known-answer seed/raw vectors | 9/9 exact |
| Fresh generator objects | Pairwise distinct |
| Second isolated trajectory | All 9 stream materials changed |
| Explicit shared namespace audit | All 9 stream materials matched |
| Auxiliary preconsumption | Core matrix/state/lineage unchanged exactly |
| Serialized request reconstruction | Exact |
| Author/MATLAB uncertainty | Preserved, unresolved |

### Example trajectory result

The canonical example is `E01-S06-SCHEMA-EXAMPLE-MODERN-GILLESPIE-v1.0.0`, a four-species modern-Gillespie fixture. It is not a paper baseline or author reconstruction.

| Field | Result |
| --- | --- |
| Requested generations | 3 |
| Logged events | 14 |
| Logged fissions | 3 |
| Terminal reason | `requested_generations_completed` |
| Lossless binary64 fields | 489 |
| Seed payload SHA-256 | `4942c2f8b5bc4a17c4a2252c7fd6a71eafb3661b4251cf1b936ccb5c69c3266d` |
| Seed file SHA-256 | `12a40d1ce3743c65a33d06b8203387ce1f26d4f5a64aef2f496362979b2c5fb1` |
| Trajectory payload SHA-256 | `aa6bc449dd5ebebf89b331c798289235e394ba1bf9af94be142d8fdefe6a9e4e` |
| Trajectory file SHA-256 | `020634d20a248ec0516040128e37aed3d7b8f5c1c6b4fde5fca7388b65b55483` |
| In-process replay | Payload/checksum/bytes exact |
| Fresh-process replay | Canonical bytes exact |

### Explicit branch coverage

| Profile | Update/time branch | Fission/daughter branch | Events | Fissions | Result |
| --- | --- | --- | ---: | ---: | --- |
| Historical distribution-comparison fixture | categorical single event / event index | fixed-size odd-discard / first | 6 | 1 | Schema, invariants, exact replay pass |
| Modern Gillespie fixture | direct Gillespie / exponential waiting | binomial complement / uniform random | 6 | 2 | Schema, invariants, exact replay pass |
| Paper-prose Poisson fixture | vector-Poisson batch / fixed exposure | binomial complement / first | 4 | 1 | Schema, invariants, exact replay pass |

This demonstrates representation and replay coverage across the existing explicit branches. It is not the goodness-of-fit or distributional validation reserved for S07.

### Registry preservation

Registry SHA-256 remained `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`. The audit found 120 parameters, 64 unresolved/conflicting/evidence-deferred items, 21 unexpanded branch sets, `executable: false`, `noSilentDefaults: true`, and zero registry updates. The trajectory embeds:

- `gard.initial_state.rng_stream = UNRESOLVED::E01-A020`
- `preprocessing.state_sampling_instant = UNRESOLVED::E01-A025`
- `authorCodeIdentity = UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND`
- `legacyMatlabRngIdentity = UNRESOLVED::LEGACY_MATLAB_RNG_ALGORITHM_AND_GLOBAL_STATE_ORDER`

Recording all event/fission state boundaries does not choose a downstream analysis sampling instant; it keeps that choice available and explicitly unresolved.

## Validation

### Validation commands

The final artifact and test runs used a fail-closed one-thread numeric environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python scripts/e01/build_rng_schema_artifacts.py --artifacts-dir /artifacts

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
pytest -q tests/e01/test_rng_schema.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
pytest -q

ruff check src scripts tests
ruff format --check src/e01_gard_reproducibility \
  scripts/e01/build_rng_schema_artifacts.py \
  scripts/e01/regenerate_s06_trajectory.py tests/e01/test_rng_schema.py

jq empty configs/e01/s06_seed_schema.json \
  configs/e01/s06_trajectory_schema.json

test ! -e /artifacts/research_steps/S07
git push origin eidosoma/groups/42
```

### Validation outcomes

- S06-focused tests: **14 passed**.
- Complete repository suite: **49 passed**.
- Ruff static check over `src`, `scripts`, and `tests`: **passed**.
- Ruff formatting check over all S06 Python files: **passed**.
- Both JSON schemas parsed and passed Draft 2020-12 meta-schema checks.
- Standalone seed envelope and complete trajectory envelope conformed.
- Custom trajectory invariants passed for the example and all three explicit S05 profiles.
- Seed and trajectory deserialization/re-serialization were byte-exact.
- A modified payload with its old checksum was rejected.
- Same-engine regeneration matched payload, checksum, and canonical bytes in-process and in a separate Python process.
- All loaded numeric thread pools reported one thread.
- S07 artifact directory was absent.
- Branch `eidosoma/groups/42` was pushed through final S06 code commit `1728b345da4b50e329c36983a436034bc25d6507`.

A repository-wide Ruff **formatting** check was explored but was not used as the S06 gate because it identified five unchanged, pre-existing S03 files that would be reformatted. Those unrelated files were not modified. The repository-wide Ruff static check passed, and the formatting check for every S06 Python file passed.

## Dependencies and runtime

- Python `3.13.14`
- NumPy `2.4.6`
- jsonschema `4.26.0`
- PCG64DXSM bit generator
- SHA-256 from the Python standard library
- PyYAML and threadpoolctl from the already frozen S03 environment
- Platform: `Linux-6.12.95+deb13-cloud-amd64-x86_64-with-glibc2.39`
- Byte order: little-endian
- Runtime fingerprint: `sha256:755207c258f156260e5854db667ae2ba2edf62ffc6a6c1e5cf06009d451a86c0`
- CPU workers: 1; loaded OpenBLAS threads: 1; GPU: not used

No dependency was installed or upgraded during S06. No network research was needed; frozen local primary/source evidence and installed runtime documentation were sufficient.

## Artifacts written

### Shared reproducibility bundle

- `/artifacts/E01_forensic_replication_bundle/reproducibility/seed_derivation_contract_v1.0.0.yaml`
- `/artifacts/E01_forensic_replication_bundle/reproducibility/trajectory_precision_contract_v1.0.0.yaml`
- `/artifacts/E01_forensic_replication_bundle/reproducibility/seed_schema_v1.0.0.json`
- `/artifacts/E01_forensic_replication_bundle/reproducibility/trajectory_schema_v1.0.0.json`
- `/artifacts/E01_forensic_replication_bundle/reproducibility/examples/example_seed_manifest.json`
- `/artifacts/E01_forensic_replication_bundle/reproducibility/examples/example_trajectory.json`
- `/artifacts/E01_forensic_replication_bundle/reproducibility/examples/example_event_record.json`
- `/artifacts/E01_forensic_replication_bundle/reproducibility/examples/example_fission_record.json`
- `/artifacts/E01_forensic_replication_bundle/reproducibility/examples/example_regeneration_manifest.json`

### S06 result artifacts

- `/artifacts/research_steps/S06/seed_validation.json`
- `/artifacts/research_steps/S06/schema_conformance.json`
- `/artifacts/research_steps/S06/branch_schema_coverage.json`
- `/artifacts/research_steps/S06/serialization_validation.json`
- `/artifacts/research_steps/S06/regeneration_validation.json`
- `/artifacts/research_steps/S06/cross_platform_precision_validation.json`
- `/artifacts/research_steps/S06/registry_preservation.json`
- `/artifacts/research_steps/S06/validation_summary.json`
- `/artifacts/research_steps/S06/artifact_manifest.json`
- `/artifacts/research_steps/S06/research_step_full_results.md`

### Git-backed reproducible code

Repository source was retained in Git, not copied into the artifact directory:

- `src/e01_gard_reproducibility/`
- `configs/e01/s06_*.yaml` and `configs/e01/s06_*.json`
- `scripts/e01/build_rng_schema_artifacts.py`
- `scripts/e01/regenerate_s06_trajectory.py`
- `tests/e01/test_rng_schema.py`

Final code identity: branch `eidosoma/groups/42`, commit `1728b345da4b50e329c36983a436034bc25d6507`. The independent-engine source aggregate SHA-256 is `6fade7fab1146fb9c0f04ab6f4062a907713414e8d3a595f5cd54cf4cf82f219`; the S06 adapter aggregate SHA-256 is `7af388bf18e259d5283ffc69f858856ffe6de614630c171f946d6c5af7d2d895`.

## Caveats, blockers, failed assumptions, and limitations

1. **No author RNG recovery.** Neither the paper nor available source evidence identifies the author seed hierarchy, MATLAB generator algorithm, stream reset order, or shared/global-state policy. S06 preserves this uncertainty instead of assigning an author-facing default.
2. **Historical behavior is different.** The public historical engine's order-sensitive legacy global RNG cannot be split into modern independent streams without altering its behavior. Explicit draw tapes remain its exact-fixture mechanism.
3. **Same-engine means identity-scoped.** A source, dependency, schema, precision, platform, byte-order, runtime, or numeric-thread identity change requires a new artifact identity; it cannot inherit the current exact-regeneration claim.
4. **Cross-platform audit is prospective.** S06 documents quantitative float bounds but did not run on a second hardware/libm platform. The current same-runtime comparison was exactly 0 ULP on all 489 captured float fields.
5. **Discrete divergence is never tolerated.** Numerical closeness does not validate a replay that selects a different event, state, fission, daughter, or stopping reason.
6. **Registry remains closed.** The example values and S05 validation profiles are explicit engineering branches. They do not resolve any of the 64 unresolved items or expand any of the 21 branch sets in registry v0.3.0.
7. **Sampling instant remains unresolved.** Capturing all boundaries avoids data loss but does not decide which boundary later scientific analyses should sample.
8. **No S07 stochastic inference.** Known-answer vectors, exact replay, and branch representation test deterministic plumbing. They do not establish distributional goodness of fit, calibrated event frequencies, lognormal moments, or fission distributions.
9. **Formatting-only non-gate.** Five unchanged S03 files do not satisfy the current Ruff formatter's style, so only S06 files were formatting-gated. All repository Python files passed Ruff static checks and all tests passed.

No scientific assumption failed inside the bounded S06 question. Two initially tempting assumptions were explicitly rejected: a seven/eight-purpose list was insufficient once the already-separated waiting-time generator was respected, and a canonical seed contract could not legitimately be applied to the historical engine as if it recovered legacy MATLAB behavior.

## Provenance

- Research date: 2026-08-01 UTC.
- Workspace root: `/workspace`.
- Repository: `/workspace/arrival-of-self-replicators`.
- Branch: `eidosoma/groups/42`.
- Final S06 code commit: `1728b345da4b50e329c36983a436034bc25d6507`.
- Original paper PDF SHA-256: `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Registry SHA-256: `aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891`.
- Runtime fingerprint: `sha256:755207c258f156260e5854db667ae2ba2edf62ffc6a6c1e5cf06009d451a86c0`.
- Seed schema SHA-256: `9eab6f1d796810fa3ba3e58d486d22293f65cac07da5c635c0aa023092e72218`.
- Final trajectory schema SHA-256: `981807b512bff589a6a693c1da191efad829ccb3294fd4f9297c3ee02a7a5d57`.
- Final precision contract SHA-256: `2c73d7385d7511636cb809cdb1b2b5c0239632faec2f6ff2ffb692a7b3548b4d`.
- Exact path/size/hash records for refreshed inputs, repository code, and outputs are in `/artifacts/research_steps/S06/artifact_manifest.json`; the manifest excludes its own hash by design.

## Final interpretation

S06 is supportive and complete. The reconstruction now has a fail-closed, versioned RNG namespace and lossless event/fission trajectory contract that exactly regenerates under the frozen S05 engine and runtime identity. This enables later stochastic validation and scientific analysis without converting unresolved author or MATLAB choices into hidden defaults. S07 remains unstarted.
