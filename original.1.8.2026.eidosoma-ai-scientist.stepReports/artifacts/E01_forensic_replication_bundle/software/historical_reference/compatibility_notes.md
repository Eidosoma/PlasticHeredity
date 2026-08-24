# S04 historical-reference compatibility notes

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S04** |
| Completion status | **Complete** for the historical-reference compatibility layer; S05 was not begun. |
| Artifacts written | Engine pointer/manifest, source traceability, compatibility matrix, 15 verified small cases, registry-preservation audit, validation summary, artifact manifest, and the canonical full-results report. |
| Validation result | **PASS** — 15/15 fixtures passed; source, registry, and API no-default checks are included in the S04 validation summary. |
| Outcome classification | **Supportive** for translating the pinned public historical behavior; constraining differences from the paper are preserved. |
| Caveats or blockers | Legacy MATLAB RNG equality is unresolved; historical GARD has no detected license; author code is unavailable; the historical source lacks paper max-steps/vector-Poisson semantics and uses different fission/initialization behavior. |
| Recommended next action | Stop after S04 and return control. If separately authorized, S05 should implement an independent engine without copying this control flow and compare only declared model-level branches. |

## Lay summary

This layer reproduces what the pinned public 2014 GARD v10 source actually does on small, explicit-draw cases. It does not say that the paper's authors used that source. Important differences—especially one-event updates, fixed-size fission, and hidden legacy random-state order—remain visible rather than being smoothed into a single “GARD” implementation.

## Compatibility matrix

| ID | Subject | Pinned historical behavior | Paper, modern, or unresolved boundary |
| --- | --- | --- | --- |
| CB01 | author implementation identity | Public GARD v10 commit 86dff6320d5a only. | Author code is unavailable; no equivalence claim is allowed. |
| CB02 | update kernel | One weighted join/leave event per loop; no vector Poisson batch and no sampled Gillespie waiting time. | Paper vector-Poisson and modern Gillespie remain distinct registry branches. |
| CB03 | time | dt accumulates total rate and orchestrator records 1/dt. | This is not physical Gillespie time; paper molecular-step semantics remain unresolved. |
| CB04 | fission | Fixed-size without-replacement child A and complement child B; odd mass discards one molecule. | Paper/plan independent Binomial(n_i,0.5) semantics remain a separate branch. |
| CB05 | daughter choice | Follow first function output child A with no extra draw. | Paper only says one daughter; author choice remains unresolved. |
| CB06 | initialization | With-replacement molecule-type sampling under hidden legacy global RNG order. | Paper says without replacement; no silent substitution is allowed. |
| CB07 | max steps | The historical growth loop has no max_steps condition. | Paper max_steps=1000 and terminal semantics remain unresolved. |
| CB08 | random streams | Legacy MATLAB global rand/randn state APIs with order-dependent resets. | Exact cross-runtime stream equality is not claimed; explicit draw tapes bypass, rather than resolve, this ambiguity. |
| CB09 | modern GARD | Historical tgs_grow_v10.m SHA-256 72cae8cd5555f4605dbad1526d3221b7f3f03eb76cea35714499299b3aa068a0. | Modern nested grow file differs and adds flux output/logging; non-drift is byte-identical only at the pinned files. |
| CB10 | license | No repository license file detected; source remains reference-in-place and is not copied into artifacts or this repository. | The independent Python port is repository-authored; this is provenance separation, not legal advice. |

## Operational rules

- All kinetic values, reservoir vectors, split sizes, matrices, and draw sources are required API inputs. There is no implicit paper profile.
- `UniformTape` is the exact fixture path. It supplies draws directly and therefore bypasses—but does not solve—legacy MATLAB RNG identity.
- `NumpyUniformSource` and `catalytic_matrix_from_numpy_rng_explicit` are explicitly labeled distribution-compatible conveniences, not legacy MATLAB stream emulators.
- The validation `event_guard` raises on exhaustion. It is not interpreted as the paper's unresolved `max_steps` terminal rule.
- The public historical source remains in `/cache` at its pinned commit and is not redistributed under artifacts or copied into the repository.
- No MATLAB or GNU Octave executable was present. Small-case validation uses hand-calculated expected values plus explicit draw tapes against the source-traced port; it does not claim original-runtime trajectory equality.

## Source identity

- Commit: `86dff6320d5ae91b4e831471079ff46749b14df9`
- Tree: `a602fc99b494982c04c60405bc6422af9db5a77a`
- Contract: `E01-historical-reference-v1.0.0`
- Engine: `1.0.0`
- License state: `NO-LICENSE-FILE-DETECTED`
- Author implementation: `UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND`
