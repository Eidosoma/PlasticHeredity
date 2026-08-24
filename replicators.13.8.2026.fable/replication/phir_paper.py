"""Paper-faithful Phi-r (Phase L; PHIR_PAPER.md). Implements the
typeset Methods page verbatim — NOT the PhiRL repository:

    Phi_r = I(X_t; X_{t+1}) - I(A_t; X_{t+1}) - I(B_t; X_{t+1})

unnormalized, parts = the two MULTIVARIATE minimum-information-
bipartition blocks (never macro-averaged), on CLR-transformed
relative per-step compositions with the last component dropped.
Registered choices forced by the page's silence: pseudocount 0.5,
diagonal ridge 1e-6*trace/dim, spectral (Fiedler) relaxation of the
instantaneous min-MI cut, per-lineage window. Gaussian throughout.
"""

from __future__ import annotations

import numpy as np

PSEUDO = 0.5
RIDGE_FRAC = 1e-6
MIN_T = 20


def clr_drop_last(counts, pseudo=PSEUDO):
    C = np.asarray(counts, dtype=np.float64) + pseudo
    rel = C / C.sum(axis=1, keepdims=True)
    L = np.log(rel)
    Z = L - L.mean(axis=1, keepdims=True)
    return Z[:, :-1]                      # the page's full-rank fix


def _ridge(S):
    d = S.shape[0]
    return S + np.eye(d) * (RIDGE_FRAC * np.trace(S) / d)


def _ld(S):
    sign, ld = np.linalg.slogdet(_ridge(S))
    return ld if sign > 0 else np.nan


def _gauss_mi(S, ia, ib):
    ia, ib = list(ia), list(ib)
    v = 0.5 * (_ld(S[np.ix_(ia, ia)]) + _ld(S[np.ix_(ib, ib)])
               - _ld(S[np.ix_(ia + ib, ia + ib)]))
    return v


def mib_instantaneous(Z):
    """Spectral relaxation of the min-MI cut on the lag-0 Gaussian MI
    graph over all retained components (registered search)."""
    r = np.clip(np.corrcoef(Z.T), -0.999999, 0.999999)
    mi = -0.5 * np.log(1 - r ** 2)
    np.fill_diagonal(mi, 0.0)
    W = mi + 1e-6
    np.fill_diagonal(W, 0.0)
    Lap = np.diag(W.sum(axis=1)) - W
    vals, vecs = np.linalg.eigh(Lap)
    f = vecs[:, 1]
    a = [i for i in range(len(f)) if f[i] > 0]
    b = [i for i in range(len(f)) if f[i] < 0]
    return a, b


def phi_r_paper(comps):
    """One paper-faithful Phi-r per lineage window."""
    comps = np.asarray(comps, dtype=np.float64)
    T = comps.shape[0]
    if T < MIN_T:
        return np.nan
    Z = clr_drop_last(comps)
    d = Z.shape[1]
    if T - 1 <= 2 * d:                    # joint covariance rank guard
        return np.nan
    A, B = mib_instantaneous(Z)
    if not A or not B:
        return np.nan
    X, Xp = Z[:-1], Z[1:]
    J = np.hstack([X, Xp])
    S = np.cov(J, rowvar=False)
    xs = list(range(d))
    xp = list(range(d, 2 * d))
    itot = _gauss_mi(S, xs, xp)
    ia = _gauss_mi(S, A, xp)
    ib = _gauss_mi(S, B, xp)
    if not (np.isfinite(itot) and np.isfinite(ia)
            and np.isfinite(ib)):
        return np.nan
    return float(itot - ia - ib)
