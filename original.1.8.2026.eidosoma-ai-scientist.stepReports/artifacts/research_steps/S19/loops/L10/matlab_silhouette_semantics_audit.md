# MATLAB-Compatible Silhouette Semantics Audit

## Concise top summary

- **Research step ID:** `S19-L10`.
- **Completion status:** source and implementation-semantics audit complete before scientific trajectory generation.
- **Artifacts written:** this audit, hashed source snapshot, implementation lock, fixture manifest/results, and MATLAB validation table.
- **Validation result:** documented singleton semantics were recovered and all 12 mandatory fixture families passed.
- **Outcome classification:** no scientific trajectory label has been opened.
- **Caveats or blockers:** the original MATLAB release and target-author code remain unavailable; only the documented singleton convention is resolved.
- **Recommended next action:** execute only the clean pushed L10 lock and return for mandatory human review.

## Direct evidence

The pinned historical GARD source at commit `86dff6320d5ae91b4e831471079ff46749b14df9` permits `k <= n`, requests MATLAB `silhouette` for multi-cluster scoring, and considers k values 1–10 with ten replicas and a four-k nonimprovement stop. Official MathWorks documentation states that a point that is the sole member of its cluster receives silhouette value 1. Official scikit-learn 1.9.0 documentation instead restricts its silhouette coefficient to `2 <= n_labels <= n_samples - 1`. Both official pages were retained cache-only and hashed in `source_snapshot_manifest.json`.

## Frozen clean-room calculation

For every non-singleton point, L10 computes `a` as mean distance to the other members of its cluster, `b` as the minimum mean distance to another cluster, and `(b-a)/max(a,b)`. A singleton receives the literal float64 value `1`. An exact `a=b=0` case receives the prospectively locked value `0`. Cosine inputs must be finite and nonzero; distance residue no smaller than `-1e-12` is clamped to zero, while a material negative distance fails closed. Cluster IDs are canonicalized by earliest assigned observation. The k=1 path remains the historical mean-H carpet and never enters the multi-cluster formula.

## Scientific recurrence boundary

Software selection and scientific recurrence are separate. A selected all-singleton solution is retained in the clustering tables but yields `NO_RECURRING_COMPTYPE`; a tied largest cluster yields `NO_UNIQUE_RECURRING_COMPTYPE`. Neither emits a molecular label or falls back to another k. This prospective gate prevents MATLAB's singleton score from fabricating a recurring compotype.

## Paper and cited-method context

The paper and Figure 1 describe molecular-composition clusters with homeostatic attractor-like growth and entry/exit relative to the most recurring composition. References 63–65 ground the GARD/composome lineage. These sources support the two registered reconstructions but do not identify the authors' exact code, MATLAB release, RNG, cluster choice, or Table 1 onset/dispersion semantics.
