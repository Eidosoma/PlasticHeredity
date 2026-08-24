"""Pointwise code-faithful Phi_R scorer for the C7 campaign
(PHIR_C7_PREREGISTRATION.md 3b).

Splits phi_r_code's pointwise machinery into a window FIT (macro
pipeline + Gaussian moments from the trailing history window) and a
batched EVALUATE at arbitrary query transitions (current state ->
candidate state), so single-molecule intervention candidates can be
scored by the Phi_R of their hypothetical transition.

Equality gate: evaluated at the window's own consecutive pairs, this
path must reproduce phi_r_code_local(window) exactly (same moments,
same lattice) — enforced by test_phi_r_point().
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from phi import clr
from phi_r_code import (ATOMS, DEAD_EPS, DESC, EPS_COV, ORDER, PHIR_ATOMS,
                        fiedler_bipartition, mi_matrix_lag1)

# variable layout for query vectors: [a_t, b_t, a_t1, b_t1]
_PAST = {(0,): [0], (1,): [1], (0, 1): [0, 1]}
_FUT = {(0,): [2], (1,): [3], (0, 1): [2, 3]}


class WindowFit:
    """Macro pipeline + Gaussian moments fitted on one history window."""

    def __init__(self, window_counts):
        window_counts = np.asarray(window_counts, dtype=np.float64)
        self.ok = False
        if window_counts.shape[0] < 20:
            return
        Zfull = clr(window_counts)                    # (T x k) CLR
        keep = Zfull.std(axis=0) > DEAD_EPS           # preprocess_data
        if keep.sum() < 3:
            return
        Z = Zfull[:, keep]
        self.ch_mu = Z.mean(axis=0)
        self.ch_sd = Z.std(axis=0)
        Zn = ((Z - self.ch_mu) / self.ch_sd).T        # channels x time
        a, b = fiedler_bipartition(mi_matrix_lag1(Zn))
        if not a or not b:
            return
        self.keep, self.a, self.b = keep, a, b
        edge = np.vstack([np.nanmean(Zn[a], axis=0),
                          np.nanmean(Zn[b], axis=0)])
        self.edge = edge
        # joint sample of (edge_t, edge_t+1): rows = [a_t, b_t, a_t1, b_t1]
        J = np.vstack([edge[:, :-1], edge[:, 1:]])    # 4 x (T-1)
        self.mu = {}
        self.cov = {}
        self.icov = {}
        self.logdet = {}
        for idx in ({0}, {1}, {2}, {3}, {0, 2}, {0, 3}, {1, 2}, {1, 3},
                    {0, 1}, {2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3},
                    {1, 2, 3}, {0, 1, 2, 3}):
            key = tuple(sorted(idx))
            x = J[list(key), :]
            self._fit_subset(key, x)
        self.ok = True

    def _fit_subset(self, key, x):
        if len(key) == 1:
            mu, sd = x[0].mean(), max(x[0].std(), 1e-12)
            self.mu[key] = mu
            self.cov[key] = sd
            return
        cov = np.cov(x, ddof=0)
        cov = cov + np.eye(cov.shape[0]) * (EPS_COV * np.trace(cov)
                                            / cov.shape[0])
        self.mu[key] = x.mean(axis=1)
        self.cov[key] = cov
        self.icov[key] = np.linalg.inv(cov)
        self.logdet[key] = np.linalg.slogdet(cov)[1]

    def macro_state(self, counts_rows):
        """(N x Ng) raw count states -> (N x 2) macro coordinates under
        the window's normalization and bipartition."""
        Zfull = clr(np.atleast_2d(np.asarray(counts_rows,
                                             dtype=np.float64)))
        Z = (Zfull[:, self.keep] - self.ch_mu) / self.ch_sd
        return np.stack([Z[:, self.a].mean(axis=1),
                         Z[:, self.b].mean(axis=1)], axis=1)

    def _h(self, key, q):
        """Pointwise Gaussian -log pdf of q[:, key] under subset moments."""
        if len(key) == 1:
            z = (q[:, key[0]] - self.mu[key]) / self.cov[key]
            return (0.5 * z ** 2 + 0.5 * np.log(2 * np.pi)
                    + np.log(self.cov[key]))
        d = q[:, list(key)] - self.mu[key]
        quad = np.einsum("ti,ij,tj->t", d, self.icov[key], d)
        return 0.5 * quad + 0.5 * (len(key) * np.log(2 * np.pi)
                                   + self.logdet[key])

    def phi_r_queries(self, q):
        """q: (N x 4) [a_t, b_t, a_t1, b_t1] -> pointwise Phi_R per row
        (identical lattice math to phi_r_code.local_phi_id/_phi_min)."""
        n = q.shape[0]
        pm = {}
        for atom in ORDER:
            i_plus = np.full(n, np.inf)
            i_minus = np.full(n, np.inf)
            for part in atom[0]:
                pk = tuple(sorted(_PAST[part]))
                i_plus = np.minimum(i_plus, self._h(pk, q))
                for tpart in atom[1]:
                    fk = tuple(sorted(_FUT[tpart]))
                    jk = tuple(sorted(_PAST[part] + _FUT[tpart]))
                    cond = self._h(jk, q) - self._h(fk, q)
                    i_minus = np.minimum(i_minus, cond)
            pm[atom] = i_plus - i_minus
        pi = {}
        for atom in ORDER:
            below = DESC[atom]
            pi[atom] = pm[atom] - (np.sum([pi[a] for a in below], axis=0)
                                   if below else 0.0)
        return np.sum([pi[a] for a in PHIR_ATOMS], axis=0)

    def score_candidates(self, current_counts, cand_counts):
        """Phi_R of the hypothetical transition current -> each candidate."""
        now = self.macro_state(current_counts[None, :])[0]
        cands = self.macro_state(cand_counts)
        q = np.hstack([np.tile(now, (len(cands), 1)), cands])
        return self.phi_r_queries(q)


def test_phi_r_point(window_counts, tol=1e-9):
    """Equality gate: fit/evaluate on the window's own pairs must equal
    phi_r_code_local(window). Returns max abs diff."""
    from phi_r_code import phi_r_code_local
    fit = WindowFit(window_counts)
    ref = np.asarray(phi_r_code_local(window_counts, "full"), float)
    if not fit.ok:
        return np.nan if np.isnan(ref).all() else np.inf
    e = fit.edge
    q = np.hstack([e[:, :-1].T, e[:, 1:].T])
    got = fit.phi_r_queries(q)
    diff = float(np.max(np.abs(got - ref)))
    assert diff <= tol, f"equality gate FAILED: max diff {diff:.3e}"
    return diff
