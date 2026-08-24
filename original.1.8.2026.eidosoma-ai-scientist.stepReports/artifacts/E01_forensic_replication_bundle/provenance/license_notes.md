# E01 S03 license notes

## Top summary

- **Research step:** S03 — Freeze source and environment snapshots
- **Completion status:** License inventory complete for the S03 source and dependency snapshot.
- **Artifacts written:** This note and `dependency_licenses.csv`; authoritative source identities are in `source_manifest.yaml`.
- **Validation result:** PASS; every frozen wheel has license metadata or an explicit unavailable marker.
- **Outcome classification:** Supportive sub-result; this is metadata capture, not legal advice.
- **Caveats or blockers:** Repository-level licensing is absent for historical GARD and absent at the modern GARD root; no redistribution is authorized by this report.
- **Recommended next action:** Keep unlicensed GARD material as commit references/cache-only inputs and consult counsel before redistribution.

## Source notes

- Paper `2607.28250v1`: CC-BY-4.0.
- `gard_historical` at `86dff6320d5ae91b4e831471079ff46749b14df9`: NO-LICENSE-FILE-DETECTED. Reference in place by commit; do not redistribute repository contents.
- `gard_modern` at `19878f6432fdfb30bea5d775175ed42a767eb3ef`: ROOT-LICENSE-NOT-DETECTED. Reference in place by commit; nested GARD_v10 files carry CC0-1.0, but no license is asserted for the repository as a whole.
- `phyid_reference` at `6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44`: BSD-3-Clause. Redistribution permitted subject to BSD-3-Clause conditions.
- `omegaid_optional` at `7fcf1fa8e288e0634f81423283d2b349ed88440e`: BSD-3-Clause. Redistribution permitted subject to BSD-3-Clause conditions.

## Dependency notes

The clean lock resolved 38 packages and `dependency_licenses.csv` records the metadata embedded in all 39 frozen wheels, including the commit-built `phyid` wheel.

License strings are transcribed from source or wheel metadata. They may be incomplete for vendored libraries and are not a legal interpretation.
