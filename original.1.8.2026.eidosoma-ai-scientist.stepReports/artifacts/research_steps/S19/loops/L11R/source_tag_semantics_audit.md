# L11R Historical Source-Tag Semantics Audit

## Concise top summary

- **Research step ID:** `S19-L11R` (`E01-S19-L11R-CENTROID-NUMERICAL-TOLERANCE-CONFIRMATION-v1.0.0`).
- **Completion status:** pre-outcome source-tag audit complete; scientific outcomes unopened.
- **Artifacts written:** this audit, paper-language audit, source snapshot, implementation lock, fixtures, pipeline registry, seed firewall, and input manifest.
- **Validation result:** all six required historical tagging statements pass direct source inspection.
- **Outcome classification:** pending; no L11R scientific label has been opened.
- **Caveats or blockers:** the public historical lineage is not author code; `Y_g=I(tag_g>0)` is a direct binary representation of the returned tag vector but the paper does not explicitly state this binary reduction.
- **Recommended next action:** commit and push the complete lock, pass the opaque benchmark, execute only U1/U2, validate, freeze, and stop.

## Source-path checks

- `TGS_NONDRIFT_RETURNS_LOGICAL_NONDRIFT_INDEX` — **PASS**; tgs_nondrift.m:40 and output documentation lines 12-13 (DIRECT_SOURCE).
- `TGS_ACLUSTER_INITIALIZES_COMPLETE_TAG_MATRIX_TO_ZERO` — **PASS**; tgs_acluster.m:45 (DIRECT_SOURCE).
- `CLUSTER_LABELS_ASSIGNED_TO_ALL_INDEXED_NONDRIFT_COMPOSITIONS` — **PASS**; tgs_acluster.m:74-76 (DIRECT_SOURCE).
- `SELECTED_CLUSTERING_RETAINS_COMPLETE_TAG_VECTOR` — **PASS**; tgs_acluster.m:84-93 (DIRECT_SOURCE).
- `HISTORICAL_SOURCE_DOES_NOT_REDUCE_BINARY_STATE_TO_LARGEST_CLUSTER` — **PASS**; source returns all selected tags and centroids; counts are calculated but no largest-cluster filter is applied (DIRECT_SOURCE).
- `SOURCE_TAG_BINARY_CAN_BE_EXPRESSED_AS_TAG_GREATER_THAN_ZERO` — **PASS**; direct inference from zero drift slots and positive one-based k-means tags (DIRECT_SOURCE_DERIVED_BINARY_INFERENCE).

## Interpretation

`tgs_nondrift.m` creates the non-drift mask. `tgs_acluster.m` allocates a zero-filled full-generation tag matrix, inserts one-based k-means tags at every non-drift position for every tested k, chooses one k by the source silhouette rule, and returns the complete selected tag vector plus every centroid. It computes cluster counts but contains no largest-cluster binary filter. Therefore `tag>0` is a source-literal drift-versus-any-compotype binary reconstruction. It is not proven to be the unavailable paper-author label.
