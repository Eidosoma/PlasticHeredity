"""Frozen reconstructed Phi-r (Phase I bridge; PHIR_BRIDGE.md).

Target paper (arXiv:2607.28250, Methods):
    Phi_r = [I(X_t;X_{t+1}) - I(A_t;X_{t+1}) - I(B_t;X_{t+1})]
            / I(X_t;X_{t+1})
on centered-log-ratio-transformed relative compositions, {A,B} the
minimum-information bipartition, Gaussian estimation. Registered
reconstruction choices documented in PHIR_BRIDGE.md. This module is
frozen before any campaign lineage runs; it is NOT the authors'
implementation and adjudicates nothing about it.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

import sim

K_SPECIES = 8
PSEUDO = 0.5
RIDGE = 1e-8
MIN_SNAPS = 20
MIN_ITOT = 1e-9
SURR_TAU_FRAC = 0.5
SURR_NOISE = 0.1


def _logdet(S):
    sign, ld = np.linalg.slogdet(S)
    if sign <= 0:
        return np.nan
    return ld


def gauss_mi(S, ia, ib):
    """I(Z[ia]; Z[ib]) for jointly Gaussian Z with covariance S."""
    ia, ib = list(ia), list(ib)
    Sa = S[np.ix_(ia, ia)]
    Sb = S[np.ix_(ib, ib)]
    Sab = S[np.ix_(ia + ib, ia + ib)]
    v = 0.5 * (_logdet(Sa) + _logdet(Sb) - _logdet(Sab))
    return v


def _bipartitions(d):
    """All bipartitions {A, B} of range(d), deterministic order."""
    idx = list(range(d))
    out = []
    for r in range(1, d // 2 + 1):
        for A in combinations(idx, r):
            if r == d - r and A[0] != 0:
                continue                 # avoid duplicate mirror splits
            B = tuple(i for i in idx if i not in A)
            out.append((list(A), list(B)))
    return out


def _mib(cov):
    """Minimum-information bipartition of components under Gaussian
    instantaneous covariance: argmin I(A;B)/min(|A|,|B|)."""
    d = cov.shape[0]
    best, best_v = None, np.inf
    for A, B in _bipartitions(d):
        v = gauss_mi(cov, A, B) / min(len(A), len(B))
        if np.isfinite(v) and v < best_v:
            best, best_v = (A, B), v
    return best


def _phi_r_joint(S, d):
    """Phi_r from a joint covariance S of [X_t (d), X_{t+1} (d)]."""
    X = list(range(d))
    Xp = list(range(d, 2 * d))
    itot = gauss_mi(S, X, Xp)
    if not np.isfinite(itot) or itot < MIN_ITOT:
        return np.nan
    part = _mib(S[:d, :d])
    if part is None:
        return np.nan
    A, B = part
    ia = gauss_mi(S, A, Xp)
    ib = gauss_mi(S, B, Xp)
    if not (np.isfinite(ia) and np.isfinite(ib)):
        return np.nan
    return float((itot - ia - ib) / itot)


def clr(counts, pseudo=PSEUDO):
    """Counts (T x k) -> relative -> centered log-ratio -> drop last
    component (the paper's full-rank fix)."""
    C = np.asarray(counts, dtype=np.float64) + pseudo
    rel = C / C.sum(axis=1, keepdims=True)
    L = np.log(rel)
    Z = L - L.mean(axis=1, keepdims=True)
    return Z[:, :-1]


def phi_r_series(comps, k=K_SPECIES):
    """Empirical Phi_r of a within-growth snapshot series.

    comps: (T x NG) composition counts. Top-k species by mean count
    (ties -> lower index), CLR, lag-1 Gaussian joint covariance,
    MIB on the instantaneous covariance, Phi_r formula.
    """
    comps = np.asarray(comps, dtype=np.float64)
    T = comps.shape[0]
    if T < MIN_SNAPS:
        return np.nan
    order = np.argsort(-comps.mean(axis=0), kind="stable")
    sel = np.sort(order[:k])
    Z = clr(comps[:, sel])
    d = Z.shape[1]
    if np.allclose(Z.std(axis=0), 0):
        return np.nan
    X, Xp = Z[:-1], Z[1:]
    J = np.hstack([X, Xp])
    S = np.cov(J, rowvar=False) + RIDGE * np.eye(2 * d)
    return _phi_r_joint(S, d)


def growth_jacobian(n, beta, sel):
    """Jacobian of the expected growth flow restricted to `sel`:
    J_ab = KF*rho*(1+beta_ab)
           - KB*(delta_ab*bn_a + n_a*(beta_ab*N - c_a)/N^2)."""
    n = np.asarray(n, dtype=np.float64)
    N = max(n.sum(), 1.0)
    c = beta @ n
    bn = 1.0 + c / N
    k = len(sel)
    J = np.empty((k, k))
    for x, a in enumerate(sel):
        for y, b in enumerate(sel):
            J[x, y] = sim.KF * sim.RHO * (1.0 + beta[a, b]) \
                - sim.KB * ((1.0 if a == b else 0.0) * bn[a]
                            + n[a] * (beta[a, b] * N - c[a]) / N ** 2)
    return J


def phi_r_surrogate(n, beta, k=K_SPECIES):
    """Selection-time surrogate: Phi_r of the one-step Gaussian
    linearization at state n (see PHIR_BRIDGE.md). Deterministic."""
    n = np.asarray(n, dtype=np.float64)
    present = np.where(n > 0)[0]
    if len(present) < 2:
        return 0.0
    order = present[np.argsort(-n[present], kind="stable")]
    sel = np.sort(order[:min(k, len(order))])
    J = growth_jacobian(n, beta, sel)
    ev = np.abs(np.linalg.eigvals(J))
    tau = SURR_TAU_FRAC / max(float(ev.max()), 1e-12)
    d = len(sel)
    A = np.eye(d) + tau * J
    top = np.hstack([np.eye(d), A.T])
    bot = np.hstack([A, A @ A.T + SURR_NOISE ** 2 * np.eye(d)])
    S = np.vstack([top, bot]) + RIDGE * np.eye(2 * d)
    # MIB on cov(X_1) (X_0 is isotropic by construction)
    X = list(range(d))
    Xp = list(range(d, 2 * d))
    itot = gauss_mi(S, X, Xp)
    if not np.isfinite(itot) or itot < MIN_ITOT:
        return 0.0
    part = _mib(S[np.ix_(Xp, Xp)])
    if part is None:
        return 0.0
    A_, B_ = part
    ia = gauss_mi(S, A_, Xp)
    ib = gauss_mi(S, B_, Xp)
    if not (np.isfinite(ia) and np.isfinite(ib)):
        return 0.0
    return float((itot - ia - ib) / itot)
