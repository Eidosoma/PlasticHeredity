# Algebraic adjudication of the paper's Φ-r identity

## Concise top summary

- **Research step ID:** `S19-L12`.
- **Completion status:** COMPLETE.
- **Artifacts written:** `phiid_atom_registry.csv`, `phirl_atom_identity_matrix.csv`, this derivation, and `metric_identity_adjudication.json`.
- **Validation result:** PASS — all 16 safe-lattice atoms were enumerated; symbolic coefficient vectors and deterministic numerical fixtures agree exactly.
- **Outcome classification:** `PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT`.
- **Caveats or blockers:** The derivation assumes the public two-source/two-target ΦID lattice; the paper does not name its redundancy convention.
- **Recommended next action:** Do not choose `integrated`, `emergence`, or direct whole-minus-parts by association strength; obtain the paper implementation or test a prospectively fixed paper-literal pipeline later.

## Lay summary

The manuscript's equation, its phrase “one atom,” and public PhiRL name three different mathematical objects. They coincide only under special cancellations that are not identities. This is a source-level discrepancy, not a failed attempt to optimize a result.

## Derivation

Let source antichains be redundancy `r`, unique source 0 `u0`, unique source 1 `u1`, and source synergy `s`; target antichains use the same names. The safe lattice contains every one of the 4×4=16 ordered atoms.

For any target antichain `q`, total whole-source information contains `r→q + u0→q + u1→q + s→q`. Information available from source part 0 contains `r→q + u0→q`; from source part 1 it contains `r→q + u1→q`. Therefore:

`I(X_t;X_t+1) - I(X_t^0;X_t+1) - I(X_t^1;X_t+1) = Σ_q (s→q - r→q)`.

Thus the displayed equation has +1 coefficients on s→r, s→s, s→u0, s→u1 and −1 coefficients on r→r, r→s, r→u0, r→u1. It is not one atom.

Public PhiRL `emergence = synergy + causation` contains only s→s, s→u0, s→u1. Public `integrated = local_phi_r` contains r→s, s→r, s→s, s→u0, s→u1, u0→s, u0→u1, u1→s, u1→u0. Neither coefficient vector equals the displayed equation's vector.

`local_phi_r` also depends on the corrected nine-atom implementation recovered in S12B/S12C; the older historical bug is preserved only as a comparator. The algebra above is independent of whether any coefficient happens to correlate with a GARD label.

## Numerical identity fixture

A deterministic vector assigning values 1 through 16 to the canonical atom order was evaluated by dot product with every registered coefficient vector. The paper equation and direct whole-minus-parts vectors agree exactly for arbitrary atoms; neither agrees identically with public `integrated` or public `emergence`. Regeneration repeats this calculation from the safe JSON, not from the pickle.
