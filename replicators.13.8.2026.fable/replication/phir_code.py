"""Code-faithful Phi-r: a deterministic port of the authors' public
implementation (github.com/pigozzif/PhiRL, information.py + main.py
compute_phi), registered in the PHIR_BRIDGE.md ADDENDUM.

Quantity: revised Phi (Phi_R) = sum of the nine 2x2 PhiID atoms with
synergy on either side plus the two cross-transfers, computed on the
two MACRO-AVERAGED halves of the Fiedler minimum-information
bipartition, with pointwise Gaussian local entropies and
pointwise-MMI redundancy, Mobius inversion on the 16-atom lattice.
Registered choices beyond their code: CLR upstream (the GARD paper's
preprocessing) and mean aggregation of the local Phi-r vector.
"""

from __future__ import annotations

import numpy as np

PSEUDO = 0.5
DEAD_EPS = 1e-8
NOISE = 1e-6
EPS_COV = 1e-6

# ---- 2x2 PhiID lattice (re-derived; checked vs phi_lattice_22) ------
R, U0, U1, S = ((0,), (1,)), ((0,),), ((1,),), ((0, 1),)
_RANK = {R: 0, U0: 1, U1: 1, S: 2}
_LEQ = {(a, b): (a == b or _RANK[a] < _RANK[b]
                 and not (_RANK[a] == 1 and _RANK[b] == 1))
        for a in (R, U0, U1, S) for b in (R, U0, U1, S)}
ATOMS = [(a, b) for a in (R, U0, U1, S) for b in (R, U0, U1, S)]
ORDER = sorted(ATOMS, key=lambda ab: _RANK[ab[0]] + _RANK[ab[1]])
DESC = {ab: [cd for cd in ATOMS
             if cd != ab and _LEQ[(cd[0], ab[0])] and
             _LEQ[(cd[1], ab[1])]]
        for ab in ATOMS}
PHIR_ATOMS = [(R, S), (U0, S), (U1, S), (S, R), (S, U0), (S, U1),
              (S, S), (U0, U1), (U1, U0)]


def _local_entropy(rows):
    """Pointwise Gaussian -log pdf (their local_entropy_1d/nd)."""
    x = np.atleast_2d(rows)
    if x.shape[0] == 1:
        mu, sd = x[0].mean(), x[0].std()
        sd = max(sd, 1e-12)
        z = (x[0] - mu) / sd
        return 0.5 * z ** 2 + 0.5 * np.log(2 * np.pi) + np.log(sd)
    cov = np.cov(x, ddof=0)
    cov = cov + np.eye(cov.shape[0]) * (EPS_COV * np.trace(cov)
                                        / cov.shape[0])
    mu = x.mean(axis=1)
    d = x.T - mu
    ic = np.linalg.inv(cov)
    sign, ld = np.linalg.slogdet(cov)
    quad = np.einsum("ti,ij,tj->t", d, ic, d)
    return 0.5 * quad + 0.5 * (x.shape[0] * np.log(2 * np.pi) + ld)


def _phi_min(edge, atom, lag=1):
    """Pointwise-MMI redundancy of one atom (their local_phi_min)."""
    n1 = edge.shape[1]
    i_plus = np.full(n1 - lag, np.inf)
    i_minus = np.full(n1 - lag, np.inf)
    for part in atom[0]:
        past = edge[list(part), :][:, :-lag]
        i_plus = np.minimum(i_plus, _local_entropy(past))
        for tpart in atom[1]:
            fut = edge[list(tpart), :][:, lag:]
            joint = np.vstack([past, fut])
            cond = _local_entropy(joint) - _local_entropy(fut)
            i_minus = np.minimum(i_minus, cond)
    return i_plus - i_minus


def local_phi_id(edge):
    """Full 16-atom pointwise PhiID via Mobius inversion."""
    pi = {}
    for atom in ORDER:
        pm = _phi_min(edge, atom)
        below = DESC[atom]
        pi[atom] = pm - (np.sum([pi[a] for a in below], axis=0)
                         if below else 0.0)
    return pi


def local_phi_r(pi):
    return np.sum([pi[a] for a in PHIR_ATOMS], axis=0)


def mi_matrix_lag1(x):
    """Their mutual_information_matrix_fast, alpha=1 (no masking)."""
    n0 = x.shape[0]
    xf, xb = x[:, :-1], x[:, 1:]
    r1 = np.corrcoef(np.concatenate([xf, xb], axis=0))[:n0, n0:]
    r2 = np.corrcoef(np.concatenate([xb, xf], axis=0))[:n0, n0:]
    r = np.clip((r1 + r2) / 2, -0.999999, 0.999999)
    mi = -0.5 * np.log(1 - r ** 2)
    np.fill_diagonal(mi, 0.0)
    return mi


def fiedler_bipartition(mi_mat, noise=NOISE):
    """Their minimum_information_bipartition: Fiedler vector of the
    weighted graph Laplacian (deterministic dense eigendecomposition;
    equality with networkx verified in the addendum record). Halves
    are the strictly positive / strictly negative entries."""
    W = mi_mat + noise
    np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(axis=1)) - W
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1]
    a = [i for i in range(len(f)) if f[i] > 0]
    b = [i for i in range(len(f)) if f[i] < 0]
    return a, b


def clr(counts, pseudo=PSEUDO):
    C = np.asarray(counts, dtype=np.float64) + pseudo
    rel = C / C.sum(axis=1, keepdims=True)
    L = np.log(rel)
    return L - L.mean(axis=1, keepdims=True)


def phi_r_code(comps):
    """Per-lineage code-faithful Phi-r (mean of the local vector).

    comps: (T x NG) per-update composition counts."""
    comps = np.asarray(comps, dtype=np.float64)
    if comps.shape[0] < 20:
        return np.nan
    Z = clr(comps).T                              # channels x time
    Z = Z[Z.std(axis=1) > DEAD_EPS]               # preprocess_data
    if Z.shape[0] < 3:
        return np.nan
    Z = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(axis=1,
                                                    keepdims=True)
    mib_a, mib_b = fiedler_bipartition(mi_matrix_lag1(Z))
    if not mib_a or not mib_b:
        return np.nan
    edge = np.vstack([np.nanmean(Z[mib_a], axis=0),
                      np.nanmean(Z[mib_b], axis=0)])
    pi = local_phi_id(edge)
    return float(np.nanmean(local_phi_r(pi)))
