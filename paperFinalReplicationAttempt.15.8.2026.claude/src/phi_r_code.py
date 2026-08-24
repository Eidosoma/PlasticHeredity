"""Code-faithful Phi_R: the quantity the authors' public implementation
(github.com/pigozzif/PhiRL, information.py compute_phi) actually computes,
as opposed to the paper's printed Psi formula (phi.py).

Algorithm (port of the PhiRL pipeline; the same port was verified against
the authors' code to ~1e-14 in the sister replication):
  1. CLR-transformed compositions (this repo's convention, phi.clr);
  2. drop dead channels (sd ~ 0), z-score the rest;
  3. pairwise lag-1 Gaussian MI matrix, symmetrized over both directions;
  4. Fiedler-vector bipartition of the weighted graph Laplacian;
  5. average the channels of each half -> two MACRO scalar series;
  6. pointwise Gaussian local entropies + pointwise-MMI redundancy,
     Mobius inversion on the 16-atom 2x2 PhiID lattice;
  7. Phi_R(t) = sum of the nine atoms with synergy on either side plus the
     two cross-transfers:
       {rts, xts, yts, str, stx, sty, sts, xty, ytx}
     = naive whole-minus-parts with the double-counted redundancy added
       back once — unnormalized, in a different sign regime from Psi.

`fit` controls the estimation window: "full" = moments from the whole
series (the authors' convention; leaks future info into early local
values), or an integer T = all estimation (dead-filter, z-score, MI
matrix, bipartition, moments) restricted to counts[:T] (leak-free).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")
from phi import clr

DEAD_EPS = 1e-8
NOISE = 1e-6
EPS_COV = 1e-6

# ---- 2x2 PhiID lattice ----------------------------------------------------
R, U0, U1, S = ((0,), (1,)), ((0,),), ((1,),), ((0, 1),)
_RANK = {R: 0, U0: 1, U1: 1, S: 2}
_LEQ = {(a, b): (a == b or _RANK[a] < _RANK[b]
                 and not (_RANK[a] == 1 and _RANK[b] == 1))
        for a in (R, U0, U1, S) for b in (R, U0, U1, S)}
ATOMS = [(a, b) for a in (R, U0, U1, S) for b in (R, U0, U1, S)]
ORDER = sorted(ATOMS, key=lambda ab: _RANK[ab[0]] + _RANK[ab[1]])
DESC = {ab: [cd for cd in ATOMS
             if cd != ab and _LEQ[(cd[0], ab[0])] and _LEQ[(cd[1], ab[1])]]
        for ab in ATOMS}
PHIR_ATOMS = [(R, S), (U0, S), (U1, S), (S, R), (S, U0), (S, U1),
              (S, S), (U0, U1), (U1, U0)]
# same nine atoms in phyid's naming, for the cross-check
PHIR_PHYID_KEYS = ["rts", "xts", "yts", "str", "stx", "sty", "sts",
                   "xty", "ytx"]


def _local_entropy(rows):
    """Pointwise Gaussian -log pdf (PhiRL local_entropy_1d/nd)."""
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
    """Pointwise-MMI redundancy of one atom (PhiRL local_phi_min)."""
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


def mi_matrix_lag1(x):
    """PhiRL mutual_information_matrix_fast, alpha=1 (no masking)."""
    n0 = x.shape[0]
    xf, xb = x[:, :-1], x[:, 1:]
    r1 = np.corrcoef(np.concatenate([xf, xb], axis=0))[:n0, n0:]
    r2 = np.corrcoef(np.concatenate([xb, xf], axis=0))[:n0, n0:]
    r = np.clip((r1 + r2) / 2, -0.999999, 0.999999)
    mi = -0.5 * np.log(1 - r ** 2)
    np.fill_diagonal(mi, 0.0)
    return mi


def fiedler_bipartition(mi_mat, noise=NOISE):
    """PhiRL minimum_information_bipartition: Fiedler vector of the
    weighted graph Laplacian; halves = strictly positive / negative."""
    W = mi_mat + noise
    np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(axis=1)) - W
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1]
    a = [i for i in range(len(f)) if f[i] > 0]
    b = [i for i in range(len(f)) if f[i] < 0]
    return a, b


def macro_halves(counts, fit="full"):
    """CLR -> preprocess -> Fiedler MIB -> two macro scalar series.

    Returns (edge 2xT, (half_a, half_b)) or (None, None) if degenerate."""
    counts = np.asarray(counts, dtype=np.float64)
    if fit != "full":
        counts = counts[:int(fit)]
    if counts.shape[0] < 20:
        return None, None
    Z = clr(counts).T                              # channels x time
    Z = Z[Z.std(axis=1) > DEAD_EPS]                # PhiRL preprocess_data
    if Z.shape[0] < 3:
        return None, None
    Z = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(axis=1, keepdims=True)
    a, b = fiedler_bipartition(mi_matrix_lag1(Z))
    if not a or not b:
        return None, None
    edge = np.vstack([np.nanmean(Z[a], axis=0), np.nanmean(Z[b], axis=0)])
    return edge, (a, b)


def phi_r_code_local(counts, fit="full"):
    """Per-step code-faithful Phi_R trajectory (length T-1, or the
    window length - 1 when fit=T). NaN array of matching length if
    degenerate."""
    edge, _ = macro_halves(counts, fit)
    T = (np.asarray(counts).shape[0] if fit == "full"
         else min(int(fit), np.asarray(counts).shape[0]))
    if edge is None:
        return np.full(max(T - 1, 0), np.nan)
    pi = local_phi_id(edge)
    return np.sum([pi[a] for a in PHIR_ATOMS], axis=0)


def phi_r_phyid_local(counts):
    """Cross-check: same nine-atom sum via phyid on the same macro pair."""
    from phyid.calculate import calc_PhiID
    edge, _ = macro_halves(counts, "full")
    if edge is None:
        return np.full(np.asarray(counts).shape[0] - 1, np.nan)
    atoms, _ = calc_PhiID(edge[0], edge[1], tau=1, kind="gaussian",
                          redundancy="MMI")
    return np.sum([np.asarray(atoms[k]) for k in PHIR_PHYID_KEYS], axis=0)
