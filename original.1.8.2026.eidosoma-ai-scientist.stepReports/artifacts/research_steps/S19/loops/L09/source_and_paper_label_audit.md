# Source and Paper Label Audit

## Concise top summary

- **Research step ID:** `S19-L09`.
- **Completion status:** source and measurement-semantics audit complete; the later locked scientific run failed closed.
- **Artifacts written:** this audit, hashed source snapshot, Table 1 semantics lock, label-method lock, fixture manifest, and source-equivalence table.
- **Validation result:** every retained source has a SHA-256 identity and 13/13 mandatory pre-outcome fixture checks passed; no source-equivalence failure triggered the later stop.
- **Outcome classification:** `LOOP_FAILED_CLOSED` for the full loop; this audit contains no scientific trajectory outcome.
- **Caveats or blockers:** no authoritative target-paper code; historical MATLAB release/RNG behavior, exact clustering details, Table 1 onset units, and SD-versus-SE identity remain unresolved.
- **Recommended next action:** preserve this audit and stop for human review; any future attempt must prospectively define the all-singleton silhouette case.

## Direct paper evidence

The paper describes recurring compositions inherited across generations, calls self-replicators clusters in molecular-composition space with homeostatic attractor-like growth, and says entry/exit depends on similarity to the run's most recurring composition. Its Methods separately describes highly similar steady compositions in Euclidean space. Figure 1 and Table 1 were treated as measurement semantics, not permission to tune a cluster radius or H threshold.

## Direct historical-source evidence

The pinned GARD v10 lineage defines H as clipped cosine similarity. Technique 1 marks a boundary non-drift when the average of incoming and outgoing adjacent-generation H exceeds 0.9, duplicating the first/last adjacent score at endpoints. `tgs_acluster` clusters only non-drift boundaries, evaluates k=1–10 with ten replicas, selects replicas by minimum distance, scores k>1 by mean silhouette, uses a special mean-H carpet score for k=1, and stops after four k values without improvement. `getcomposometime_v10` and `biased_gard_v10` identify the most frequent compotype.

## Reconstruction choices

R1 used deterministic CPU-float64 spherical k-means because the original MATLAB release and RNG are not identified. R2 followed the paper's Euclidean wording with deterministic Lloyd k-means. R2's k=1 silhouette was explicitly undefined and could not win selection. Both pipelines required at least two assigned members and two strict-H>0.9 centroid visits. These are frozen reconstructions, not author-code claims.

The later real-input failure exposed a choice the lock had not defined for R1: when historical filtering leaves n points and the k search reaches k=n, all clusters are singletons and the selected backend does not define silhouette. No post-outcome convention was added.

## Cited-method context

References 63–65 ground the GARD/composome lineage. The open PNAS text defines H and homeostatic quasi-stationary composomes; related GARD papers describe compotypes and compositional recurrence. Public identities, retrieval paths, hashes, and license/redistribution status are in `source_snapshot_manifest.json` and the append-only source ledger. No author was contacted, and unlicensed source was not redistributed.

## Table 1 semantics

Molecular probability, persistence, consecutive-label Pearson consistency, and onset were frozen in `table1_semantics_lock.yaml`. Zero-based, one-based, normalized, and fission-generation onset were all required. Both sample SD and SE were required; `AUTHOR_DISPERSION_UNRESOLVED` could not be resolved by target proximity. Boundary diagnostics could not replace molecular results.

## Reporting amendment boundary

`S19-L09-REPORTING-AMENDMENT-001` restored this intended standalone audit after a Markdown serialization defect. It changed no source identity, scientific method, value, failure, or classification.
