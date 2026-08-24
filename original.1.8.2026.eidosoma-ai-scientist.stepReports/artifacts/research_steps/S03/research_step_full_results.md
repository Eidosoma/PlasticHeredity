# S03 full results — Freeze source and environment snapshots

## Top summary

- **Research step:** S03 — Freeze source and environment snapshots
- **Completion status:** Complete; S03 only. S04 was not begun.
- **Artifacts written:** Canonical source manifest and source-file hashes; immutable commit verification and author-code search; source-evidence specification registry v0.3.0 and delta audit; compiled, installed-Python, and system locks; 39-wheel dependency/hash/license inventory; runtime/environment and precision reports; clean-environment smoke result; validation summary; artifact manifest; and repository code/config/tests.
- **Validation result:** PASS. The official paper and four repositories passed hash/commit/tree/ref/file/archive checks; 38 locked packages and 39 dependency wheels (including the commit-built `phyid` wheel) were frozen; an isolated Python 3.13.14 environment passed CPU/GPU tests; the six authorized registry items changed with 114/114 non-source parameters preserved; 13/13 repository tests and Ruff passed; prior S01/S02 outputs remained intact; and no S04 artifact directory exists.
- **Outcome classification:** **Supportive** for the bounded S03 hypothesis that the reconstruction inputs and runnable environment can be frozen immutably despite external changes. This is not a replication verdict for any paper result.
- **Caveats or blockers:** No author-code release was found. The exact paper-to-`phyid` atom/aggregate mapping and author redundancy choice remain unresolved. The parent OCI reference/digest is not exposed and is represented by explicit unavailable sentinels plus a composite runtime fingerprint. Historical GARD has no detected license and modern GARD has no root license. Two L4 GPUs were visible although the generic runtime description mentions one.
- **Lay summary:** The paper PDF, reference programs, and every Python package needed for this stage now have exact version or commit identities and hashes. A fresh Python installation reproduced the scientific stack and passed both CPU and GPU checks without patching. Where the sources still do not say what the authors chose, the registry continues to say “unresolved” rather than guessing.
- **Recommended next action:** Hand control back. If the Chief Scientist separately authorizes S04, build only from the pinned historical GARD commit, keep the modern/historical file difference explicit, select GPU hardware by UUID, and preserve all remaining unresolved sentinels and branch sets.

## Frozen question

Can the reconstruction be made immutable and rerunnable despite external repository changes?

The prospective S03 success criterion was that every required source and dependency receive an immutable identity, the available runtime and numerical precision be recorded without silent defaults, commits be verified, and a clean environment pass a meaningful smoke test. The criterion was met with one explicit infrastructure limitation: the parent OCI image digest is not visible from inside the workspace, so it was not invented or inferred.

## Inputs

### Governing and prior-step inputs

- `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and the pre-S03 `/workspace/RESEARCH_PLAN.md`.
- `/workspace/input-attachments/MANIFEST.json`, the required `_metadata/ATTACHMENT.md` sidecar, and the complete Docling paper extraction.
- S01 full results, 59-row claim ledger, 12-row source reconciliation, and S01 provenance/validation artifacts.
- S02 full results, 105-row ambiguity ledger, 12-row discrepancy taxonomy, 59-row claim crosswalk, 120-parameter registry v0.2.0, and S02 provenance/validation artifacts.
- The official arXiv v1 record and 1,117,911-byte equation-bearing PDF for arXiv `2607.28250v1`.
- The historical GARD, modern GARD, authoritative `phyid`, and optional ΩID repositories named by the plans.
- Runtime capability records. No registered scientific capability wrapper was required and no dataset was required for S03.

The supplied attachment manifest said the original PDF was not materialized but recorded its original size as 1,117,911 bytes. The official arXiv v1 PDF has exactly that size and SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`. Size agreement is supporting reconciliation evidence; the official URL, version, and cryptographic hash are the actual source identity.

## Detailed methods

### 1. Source discovery and immutable pinning

The official arXiv record was queried by identifier, its v1 PDF was downloaded, hashed, converted with `pdftotext -layout`, and page 7 was visually checked. The e-print/source endpoint returned the same PDF bytes, consistent with a PDF-only submission. Repository sources were cloned to `/cache/e01_s03/sources/`, checked out detached at selected commits, and recorded by remote URL, ref, commit, tree, commit date, archive hash, and selected file hashes. `git fsck --full --strict` and remote-ref checks were required.

Locally generated `git archive --format=tar.gz` snapshots were kept under `/cache/e01_s03/archives/`, not under collectible artifacts. Commit/tree identities are canonical; compressed archive byte hashes are additional snapshot evidence and may depend on the exact archive producer.

Author-code discovery checked the arXiv record/code-link state, exact-title and GARD-related GitHub repository searches, and all 34 public repositories visible for author account `pigozzif` on 2026-08-01. No identifiable code release for this paper was found. This is a dated absence result, not a claim that code can never appear.

### 2. Source-semantic audit

The official paper PDF recovered two equations missing from the supplied extraction:

\[
\Phi^r = I(X_t, X_{t+1}) - \sum_i I(X_t^i, X_{t+1})
\]

and the centered log-ratio transform described on the same page,

\[
X_t = \log\!\left(\frac{X_t}{\operatorname{geometric\ mean}(X)}\right).
\]

The pinned `phyid` source returns 16 atom keys (`rtr` through `sts`), accepts Gaussian or discrete estimators, supports MMI and CCS redundancy functions, and uses Gaussian/MMI as code defaults. It does not define a named `Phi^r` aggregate matching the paper expression. Therefore the equation-source item could be resolved, but the atom mapping and author redundancy choice could not. A library default was not promoted into an author-method claim.

Historical and modern GARD copies were compared by hash. `tgs_grow_v10.m` differs between repositories, while `tgs_nondrift.m` is byte-identical. The source manifest preserves both facts and forbids silent substitution or generalizing one matching file to repository equivalence.

### 3. Dependency locking and artifact hashing

The required Python stack was frozen for CPython 3.13 on x86-64 manylinux with the CUDA 12.8 PyTorch backend. `uv pip compile --generate-hashes` produced a 38-package transitive lock. Every selected wheel was downloaded with pip `--require-hashes` into `/cache/e01_s03/wheelhouse/`. The selected wheel filename, version, size, SHA-256, Python requirement, and embedded license metadata were written to `dependency_artifacts.csv` and `dependency_licenses.csv`.

`phyid` has no released wheel tied to the selected commit. A wheel was built from the detached, hash-verified source at commit `6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44`; its version is `0+untagged.8.g6c5f2e9` and its wheel SHA-256 is `374aa40ac4591b294e50c10c5bf5e71f1c2f8ebc59e5398374320817db93997c`. That wheel is independently hash-pinned in the dependency table and source manifest rather than falsely represented as a registry-index artifact.

### 4. Runtime, hardware, and precision capture

The environment capture records OS/kernel/glibc, Python/pip/uv, hashes of key runtime binaries, installed-distribution RECORD/METADATA hashes, the complete base Python freeze, the clean-environment freeze, system packages, CPU availability/cpuset, both visible GPU identities, driver, compute capability, CUDA/cuDNN state, and a precision policy.

The parent OCI image is outside the nested Docker daemon. `docker inspect` could not resolve the parent container and the inner daemon listed no parent image. Rather than infer a digest from an overlay path, the report uses:

- `UNAVAILABLE::PARENT_IMAGE_REFERENCE_NOT_EXPOSED`
- `UNAVAILABLE::PARENT_OCI_DIGEST_NOT_EXPOSED`
- composite runtime fingerprint `sha256:755207c258f156260e5854db667ae2ba2edf62ffc6a6c1e5cf06009d451a86c0`

The composite binds the available OS, kernel, key binaries, system/Python locks, and installed core distribution records. It is explicitly not an OCI manifest digest.

The numerical reference policy is CPU float64. GPU validation requires an explicit index and UUID, float64, TF32 disabled, deterministic validation algorithms, and 1e-10 absolute/relative tolerance. Float32 is permitted only under an explicitly named future mode. ΩID stays on the NumPy backend in the frozen core environment; CuPy acceleration is a separate optional environment.

### 5. Clean Python 3.13 environment smoke

A new virtual environment was created under `/cache/e01_s03/smoke/.venv` from `/opt/python/3.13.14/bin/python3.13`. The 38-package lock was installed offline from the hash-verified wheelhouse. The separately hash-checked `phyid` wheel was installed with dependencies disabled, then `uv pip check` verified the 39-package installed environment.

The smoke exercised:

- NumPy float64 calculations;
- SciPy Spearman correlation;
- scikit-learn linear regression;
- a compiled Numba kernel;
- both MMI and CCS calculations in pinned `phyid`;
- both MMI and CCS calculations in ΩID using its NumPy backend;
- PyTorch float64 CPU matrix multiplication;
- PyTorch float64 on explicit `cuda:0`, UUID `GPU-1f5bed19-d63f-be98-470c-f670e3d4aafd`, with TF32 disabled and deterministic algorithms enabled;
- CPU/GPU comparison at the frozen tolerance.

One worker/thread was used for the smoke (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMBA_NUM_THREADS` all set to 1). Source and registry validation were serial. The project-wide maximum remains eight CPU workers for later steps.

### 6. Source-only registry update

Registry v0.2.0 was treated as immutable S02 evidence. S03 wrote a distinct versioned file, `specification_registry_v0.3.0.yaml`; it did not overwrite the S02 registry or ledger. Only the six S03-owned ambiguity IDs were authorized for source-evidence updates:

- `E01-A001`, paper formula source: resolved to official arXiv v1 PDF.
- `E01-A002`, author code: retained exact unresolved sentinel.
- `E01-A003`, historical GARD revision: resolved to pinned commit.
- `E01-A004`, authoritative `phyid` revision: resolved to pinned commit.
- `E01-A043`, `Phi^r` atom identity/formula: retained unresolved because the formula is known but the implementation mapping is not.
- `E01-A044`, redundancy function: retained unresolved because MMI default and CCS support do not establish the author choice.

The update validator compared all other registry entries by parameter, not merely by ambiguity ID, which also covers the 15 known fixed parameters that have no ambiguity ID. All 114 non-source parameters were byte-semantically unchanged. Existing conflicts, branch sets, and exact unresolved sentinels were preserved.

## Commands

Key reproducible commands, run from `/workspace/arrival-of-self-replicators`, were:

```bash
python scripts/e01/build_source_snapshot.py

uv pip compile configs/e01/s03_requirements.in \
  --python-version 3.13 \
  --python-platform x86_64-manylinux_2_28 \
  --torch-backend cu128 \
  --generate-hashes \
  --output-file /artifacts/E01_forensic_replication_bundle/provenance/requirements-s03-py313-cu128.lock

python -m pip download \
  --dest /cache/e01_s03/wheelhouse \
  --require-hashes --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r /artifacts/E01_forensic_replication_bundle/provenance/requirements-s03-py313-cu128.lock

uv build --wheel \
  --out-dir /cache/e01_s03/phyid-wheel \
  /cache/e01_s03/sources/phyid

uv venv --python /opt/python/3.13.14/bin/python3.13 \
  /cache/e01_s03/smoke/.venv

uv pip sync \
  --python /cache/e01_s03/smoke/.venv/bin/python \
  --require-hashes --no-index \
  --find-links /cache/e01_s03/wheelhouse \
  /artifacts/E01_forensic_replication_bundle/provenance/requirements-s03-py313-cu128.lock

uv pip install \
  --python /cache/e01_s03/smoke/.venv/bin/python \
  --no-index --no-deps \
  /cache/e01_s03/phyid-wheel/phyid-0+untagged.8.g6c5f2e9-py3-none-any.whl

uv pip check --python /cache/e01_s03/smoke/.venv/bin/python

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /cache/e01_s03/smoke/.venv/bin/python \
  scripts/e01/s03_clean_smoke.py \
  --device-index 0 \
  --output /artifacts/research_steps/S03/clean_environment_smoke.json

python scripts/e01/capture_environment_snapshot.py
ruff check scripts/e01 tests/e01 configs/e01
pytest -q
git diff --check
python scripts/e01/validate_s03_artifacts.py --manifest
```

Repository changes were committed as `97ed4cd58a13f8a0145f97f629de2e1bfaf8a0d9` and pushed to `origin/eidosoma/groups/42`; the remote ref was checked for exact equality.

No sudo, OS package installation, ad hoc Python installation, Docker-in-Docker workload, simulation run, or S04 command was used.

## Results

### Immutable source identities

| Source | Immutable identity | Tree or file SHA-256 | License note | Verification |
| --- | --- | --- | --- | --- |
| Official paper | arXiv `2607.28250v1`; PDF SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4` | 1,117,911 bytes; equation-bearing page 7 | CC-BY-4.0 | PASS |
| Historical GARD | commit `86dff6320d5ae91b4e831471079ff46749b14df9` | tree `a602fc99b494982c04c60405bc6422af9db5a77a` | No license file detected; reference-only | PASS |
| Modern GARD | commit `19878f6432fdfb30bea5d775175ed42a767eb3ef` | tree `b6655121074f7fdf4474f270e49bc74045e5de53` | No root license; nested GARD v10 is CC0-1.0 only | PASS |
| `phyid` | commit `6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44` | tree `fdfe5a21190062b9dda7c8831f72438d8ff5ea95` | BSD-3-Clause | PASS |
| Optional ΩID | release tag `v0.2.5`, commit `7fcf1fa8e288e0634f81423283d2b349ed88440e` | tree `33ab6f59592048e78a691ecffd9a3dff6d95e54d` | BSD-3-Clause | PASS |
| Author code | `UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND` at dated search | Search provenance in JSON | Paper promises later availability | Unresolved, explicitly preserved |

All repository commits exist locally as commit objects, have the expected trees, pass strict `git fsck`, match the selected remote refs (including peeled ΩID tag), and have verified required-file and local-archive hashes.

### Required dependency pins

| Dependency | Version | Selected wheel/source SHA-256 |
| --- | --- | --- |
| NumPy | 2.4.6 | `a7830bab239b79cda9c08c2da014761cafb48da6150e1da17ac06283f43b6089` |
| SciPy | 1.18.0 | `a46f9273dbd0eb1cefba61c9b8648b4dfe3cbc14a080176f9a73e44b8336dc7f` |
| scikit-learn | 1.9.0 | `da76d09304a4706db7cc1e3ebaa3b6b98a67365cc11d2996c4f1e58ba47df714` |
| Numba | 0.65.1 | `c09f49117ef255e1f1c6dad0c7a1ed39868243862a73be5706793241a3755f1b` |
| PyTorch CUDA 12.8 | 2.11.0+cu128 | `db964b33c55035a72ab3e2162287af8f1cc276039c65d015740cc88c26dcedf7` |
| ΩID | 0.2.5 | `8077bd698f53aa881c6728bc56d6d20527c0fcb51fd9e4d1a9ed3b052cbee632` |
| `phyid` commit build | 0+untagged.8.g6c5f2e9 | `374aa40ac4591b294e50c10c5bf5e71f1c2f8ebc59e5398374320817db93997c` |

The full closure contains 38 locked registry packages and 39 selected wheels after adding the commit-built `phyid` wheel. Total selected wheel bytes are 4,076,052,959 and remain under `/cache`, while compact hashes and metadata are collectible artifacts. Compiled lock SHA-256 is `69ca2aaa24ae90e129d3f5356520ad5621b7c9049902c5df588418a3c0f54d7f`.

### Runtime and hardware

- Ubuntu 24.04.1 LTS; kernel `6.12.95+deb13-cloud-amd64`; glibc 2.39; x86-64.
- CPython 3.13.14; pip 26.1.2; uv 0.11.29.
- CUDA toolkit/runtime target 12.8.1; PyTorch compiled CUDA 12.8; driver 610.43.02; cuDNN 9.19.0 as reported by the clean PyTorch environment.
- Cgroup-visible CPUs: 24 logical CPUs (`0-23`); E01 policy maximum: 8 workers.
- Visible GPUs: two NVIDIA L4 devices, compute capability 8.9, 23,034 MiB each:
  - `GPU-1f5bed19-d63f-be98-470c-f670e3d4aafd`
  - `GPU-971a1bac-d6a8-ecb8-6880-4fc4728f2220`
- Smoke device: explicit `cuda:0`, first UUID above.

The preloaded base process had cuDNN TF32 enabled at capture, while the frozen validation policy and clean smoke explicitly disabled both cuDNN and matrix-multiply TF32. The distinction is recorded to prevent base-library state from becoming a silent numerical default.

### Clean smoke metrics

- Virtual-environment isolation: true; Python 3.13.14.
- Package consistency: 39 packages compatible.
- NumPy/Numba sum-of-squares absolute difference: 0.
- SciPy monotonic Spearman rho: 1.0.
- scikit-learn deterministic linear-regression (R^2): 1.0.
- `phyid`: 16 finite atoms for MMI and 16 finite atoms for CCS.
- ΩID/NumPy backend: 16 finite atoms for MMI and 16 finite atoms for CCS.
- PyTorch CPU/GPU dtype: float64.
- Maximum CPU/GPU absolute and relative difference: 0 in the frozen fixture.
- GPU TF32: disabled; deterministic algorithms: enabled.
- Python 3.13 compatibility patch applied: false.

### Registry result

| Measure | v0.2.0 before S03 | v0.3.0 after S03 |
| --- | ---: | ---: |
| Parameters | 120 | 120 |
| Unresolved/conflict/evidence-deferred | 67 | 64 |
| Unexpanded branch sets | 21 | 21 |
| Total execution blockers | 88 | 85 |
| Execution gate open | No | No |
| No-silent-default flag | Yes | Yes |

Three source identity items were resolved, three remained exact unresolved sentinels, six total S03-owned entries received source-evidence updates, zero out-of-scope entries changed, and 114 non-source parameters were preserved.

## Validation

Validation layers and outcomes:

1. **Source integrity:** official PDF size/hash; four repository commit/tree/object/ref checks; strict Git fsck; four local archive hashes; required source-file hashes. PASS.
2. **Dependency integrity:** 38 versions in a generated hash lock; 38 selected registry wheels with hashes present in the lock; one commit-built `phyid` wheel with independent hash; 39 unique package artifacts. PASS.
3. **Environment isolation:** fresh CPython 3.13.14 venv, offline hash-enforced sync, `uv pip check`, CPU scientific stack, both PhiID redundancy paths, explicit float64 GPU comparison. PASS.
4. **Precision contract:** explicit CPU/GPU dtype, UUID, TF32, determinism, tolerance, and thread policy; no silent precision changes. PASS.
5. **Registry contract:** expected v0.2.0 lineage hash, six allowed IDs only, three source resolutions only, exact remaining sentinels, 114/114 non-source parameters unchanged, conflicts/branches preserved, gate still closed. PASS.
6. **Repository validation:** `ruff check scripts/e01 tests/e01 configs/e01`, 13/13 pytest tests, `git diff --check`, focused commit, push, and remote branch equality. PASS.
7. **Prior evidence integrity:** S01 and S02 output-role hashes still match their manifests. PASS.
8. **Scope boundary:** `/artifacts/research_steps/S04` absent. PASS.

The canonical machine-readable outcomes are `validation_summary.json`, `commit_verification.json`, `source_registry_validation.json`, and `clean_environment_smoke.json`.

## Failed attempts and recoveries

- The first compiled lock intentionally omitted setuptools as apparent build tooling. Hash-enforced download correctly failed because PyTorch 2.11 declares runtime dependency `setuptools<82`. The lock was regenerated with `setuptools==81.0.0` and hashes; the second download succeeded. No partial environment was accepted.
- The first venv command used the plan-like path `/opt/python/3.13.14/bin/python`, which does not exist in this image. The actual executable is `/opt/python/3.13.14/bin/python3.13`. The failed attempt created no venv or packages; the corrected command used fail-fast shell semantics and passed.
- Initial wheel metadata inspection assumed a wheel contained one `.dist-info/METADATA`; setuptools vendors several. Validation was tightened to select the single top-level distribution metadata while retaining vendored contents in the wheel hash.
- `phyid` built on Python 3.13 without code changes but emitted an upstream packaging warning because `pyproject.toml` names missing `README.md` while the repository contains `README.rst`, plus future setuptools license-metadata deprecation warnings. These did not affect import or numerical smoke behavior and were not patched silently.

## Artifacts written

### Shared E01 provenance and specifications

- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/source_manifest.yaml`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/source_file_hashes.csv`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/environment_report.json`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/environment_report.md`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/requirements-s03-py313-cu128.lock`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/clean_environment_python_freeze.txt`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/base_environment_python_freeze.txt`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/system_packages.lock`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/dependency_artifacts.csv`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/dependency_licenses.csv`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/precision_policy.yaml`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/license_notes.md`
- `$ARTIFACTS_DIR/E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml`

The S02 registry at `specification_registry.yaml` remains unchanged and hash-valid.

### S03 handoff directory

- `$ARTIFACTS_DIR/research_steps/S03/research_step_full_results.md`
- `$ARTIFACTS_DIR/research_steps/S03/commit_verification.json`
- `$ARTIFACTS_DIR/research_steps/S03/author_code_search.json`
- `$ARTIFACTS_DIR/research_steps/S03/registry_update_audit.csv`
- `$ARTIFACTS_DIR/research_steps/S03/source_registry_validation.json`
- `$ARTIFACTS_DIR/research_steps/S03/clean_environment_smoke.json`
- `$ARTIFACTS_DIR/research_steps/S03/validation_summary.json`
- `$ARTIFACTS_DIR/research_steps/S03/artifact_manifest.json`

### Repository source

- `configs/e01/s03_source_pins.yaml`
- `configs/e01/s03_registry_updates.yaml`
- `configs/e01/s03_requirements.in`
- `configs/e01/s03_precision_policy.yaml`
- `scripts/e01/build_source_snapshot.py`
- `scripts/e01/capture_environment_snapshot.py`
- `scripts/e01/s03_clean_smoke.py`
- `scripts/e01/validate_s03_artifacts.py`
- `tests/e01/test_source_snapshot.py`
- `tests/e01/test_environment_snapshot.py`

Repository commit: `97ed4cd58a13f8a0145f97f629de2e1bfaf8a0d9`, pushed to `eidosoma/groups/42`.

## Caveats, blockers, and limitations

- **No author code:** the promised release was not located at the dated search locations. Related repositories cannot be relabeled as the paper implementation.
- **Formula is not implementation identity:** recovering the scalar equation does not identify which `phyid` atom or aggregate was used.
- **Redundancy remains unresolved:** `phyid` default MMI is not proof of the authors’ choice; MMI and CCS remain explicit candidates.
- **Registry remains non-executable:** 64 unresolved items and 21 unexpanded branch sets remain. S03 did not authorize resolving non-source methods.
- **OCI digest unavailable:** the composite fingerprint improves rerunnability but cannot substitute for an OCI manifest digest. The sentinel must remain until infrastructure supplies the actual reference/digest.
- **Hardware visibility differs from generic plan:** two L4s were visible. Future run manifests must bind an explicit index and UUID; they may not rely on “first GPU” as an unstated default.
- **Licensing:** no redistribution permission is inferred for historical GARD or the modern repository as a whole. The nested modern GARD v10 CC0 notice is not generalized to the repository root. Wheel license strings are metadata capture, not legal advice.
- **Archive byte identity:** local `git archive` gzip hashes are snapshot evidence; commit and tree hashes are the durable Git identities.
- **Base image breadth:** the full preinstalled environment is larger than the clean lock. Later work should use the clean lock or explicitly record any additional dependency rather than rely on incidental base packages.
- **Scientific boundary:** S03 tests software availability and numerical plumbing only. It does not validate the GARD engine, Phi-r estimator correctness for the paper, causal claims, or any reported effect.

## Provenance

- Date: 2026-08-01 UTC.
- Workspace: `/workspace`; repository: `/workspace/arrival-of-self-replicators`.
- Branch: `eidosoma/groups/42`; immutable source-code commit: `97ed4cd58a13f8a0145f97f629de2e1bfaf8a0d9`; remote equality verified after push.
- Artifact root: `/artifacts`; disposable sources, archives, wheels, and venv: `/cache/e01_s03`.
- Source identities, retrieval URLs, ref/commit/tree dates, file/archive/dependency hashes, license notes, environment locks, and runtime fingerprint are machine-readable in the source manifest and environment report.
- The official paper is arXiv `2607.28250v1`, published 2026-07-30, PDF SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- S01/S02 ledgers, reports, source reconciliation, discrepancy taxonomy, and registry v0.2.0 were read as inputs and not overwritten.
- `RESEARCH_PLAN.md` was updated after S03 to show no active step, S03 complete, S04 queued but unstarted, and the current caveats/recommendation.

Control returns here. No S04 work was performed.
