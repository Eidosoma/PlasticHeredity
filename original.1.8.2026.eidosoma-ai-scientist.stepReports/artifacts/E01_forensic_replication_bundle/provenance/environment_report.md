# E01 S03 environment report

## Top summary

- **Research step:** S03 — Freeze source and environment snapshots
- **Completion status:** Environment capture complete.
- **Artifacts written:** JSON environment report, compiled and installed locks, system-package lock, dependency/hash/license tables, precision policy, and clean smoke result.
- **Validation result:** PASS; 38 locked packages, 39 frozen wheels including `phyid`, and clean Python 3.13 CPU/GPU smoke success.
- **Outcome classification:** Supportive sub-result.
- **Caveats or blockers:** Parent OCI image digest is not exposed and remains an explicit unavailable sentinel; the composite runtime fingerprint is not an OCI digest. Two L4 GPUs were visible although the generic plan described one fast GPU.
- **Recommended next action:** Use the compiled lock and explicit device UUID/precision fields for later environments; do not infer an image digest or GPU selection.

## Runtime and image identity

- OS: Linux-6.12.95+deb13-cloud-amd64-x86_64-with-glibc2.39
- Python: 3.13.14 at `/opt/python/3.13.14/bin/python3.13`
- CUDA environment: 12.8.1; PyTorch: 2.11.0+cu128
- OCI image digest: `UNAVAILABLE::PARENT_OCI_DIGEST_NOT_EXPOSED`
- Composite runtime fingerprint: `sha256:755207c258f156260e5854db667ae2ba2edf62ffc6a6c1e5cf06009d451a86c0`

The parent image is outside the nested daemon and its OCI reference/digest is not visible. The fingerprint binds the available OS, kernel, key-binary hashes, system/Python locks, and installed core distribution records without pretending to be an OCI manifest identity.

## CPU, GPU, and precision

- CPU: Intel Xeon (see lscpu JSON); cgroup exposes 24 logical CPUs, while project policy permits at most 8 workers.
- Visible GPUs: cuda:0 NVIDIA L4 GPU-1f5bed19-d63f-be98-470c-f670e3d4aafd, cuda:1 NVIDIA L4 GPU-971a1bac-d6a8-ecb8-6880-4fc4728f2220
- Frozen numerical reference: CPU float64.
- GPU validation: explicit device index and UUID, float64, TF32 disabled, deterministic validation algorithms, and 1e-10 absolute/relative cross-device tolerance.
- ΩID default backend: NumPy; CuPy is not in the frozen core environment and cannot be selected implicitly.

## Locks and clean-environment result

- Compiled lock SHA-256: `69ca2aaa24ae90e129d3f5356520ad5621b7c9049902c5df588418a3c0f54d7f`
- Clean freeze SHA-256: `1f01a8035575a51d376db96eaa071bb1d1f021ae0e64885ee4f33a7286301893`
- System lock SHA-256: `f06aeb6ae304517a6430c83a63c651aa8465c5f1785094f529c17dc652c4b6e8`
- Clean environment: Python 3.13.14, virtual environment `/cache/e01_s03/smoke/.venv`, success `true`.
- Python 3.13 patching: none. Released wheels installed directly; the pinned `phyid` commit built with a missing-README metadata warning but no source modification.

## Validation

All resolved wheel hashes are present in the compiled lock: `true`. Dependency consistency and the CPU/GPU smoke passed. Validation errors: `none`.
