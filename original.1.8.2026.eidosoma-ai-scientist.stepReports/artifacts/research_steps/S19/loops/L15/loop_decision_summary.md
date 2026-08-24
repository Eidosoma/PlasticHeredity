# S19-L15 Decision Summary

**Status:** complete; mandatory human review required.
**Primary classification:** `EXPLORATORY_NON_SUPPORT`.
**Additional classifications:** `EXPLORATORY_NON_SUPPORT, POSSIBLE_PIPELINE_ARTIFACT, NOT_PROMOTABLE`.

The new 200-matrix paired run executed the full S16 all-cell padding mechanism that L14 never opened.

- CANDIDATE_2: 199/200 eligible; q=0.5883; real-label prevalence=0.9829; padded prevalence=0.5782; padded dummy=0.5782.
- CANDIDATE_3: 199/200 eligible; q=0.6200; real-label prevalence=0.9837; padded prevalence=0.6099; padded dummy=0.6099.

- CANDIDATE_2: P1_PHIRL_EMERGENCE_COMPLETED_FIT=0.9786 (paper≈0.8485); B1_COMPOSITION_CHANGE=0.9786 (paper≈0.8054); B2_RAW_COMPOSITIONS=0.9796 (paper≈0.7992); B3_MOLECULAR_FLUXES=0.9810 (paper≈0.7900); D0_MAJORITY_DUMMY=0.5836 (paper≈0.6092).
- CANDIDATE_3: P1_PHIRL_EMERGENCE_COMPLETED_FIT=0.9784 (paper≈0.8485); B1_COMPOSITION_CHANGE=0.9782 (paper≈0.8054); B2_RAW_COMPOSITIONS=0.9793 (paper≈0.7992); B3_MOLECULAR_FLUXES=0.9797 (paper≈0.7900); D0_MAJORITY_DUMMY=0.6184 (paper≈0.6092).

Padding-dominance status: `True`. Exact full-panel gate in both candidates: `False`. Directional full-panel gate in both candidates: `False`.

This is forensic only. Return for mandatory human review; no L16, S20, E02, author contact, intervention, or report bundle is active.
