"""Phase F infrastructure: Singh–Jain multistable protocell
(arXiv:2310.19744) — the analytic bistable POSITIVE CONTROL for the
basin assay.

Species: A(1) monomer X1, A(2) dimer X2, A(4) tetramer/catalyst X4.
Volume V = X1 + 2*X2 + 4*X4 (v = 1). Dimensionless parameters from the
paper: kF = 1, kR = 1, phi = 20, alpha = 100; bistable for
kappa in (1840, 3580); reference kappa = 2400; division at V >= Vc =
1000 with independent binomial(0.5) partitioning, one daughter tracked.

Stochastic propensities (Gillespie; combinatorics X(X-1) for
bimolecular; the paper's deterministic limit is recovered for large X):
  transport  A1ext -> A1 :      alpha * X2
  R1 fwd     2A1 -> A2   :      kF * X1*(X1-1)/V * (1 + kappa*X4/V)
  R1 rev     A2 -> 2A1   :      kR * X2        * (1 + kappa*X4/V)
  R2 fwd     2A2 -> A4   :      kF * X2*(X2-1)/V * (1 + kappa*X4/V)
  R2 rev     A4 -> 2A2   :      kR * X4        * (1 + kappa*X4/V)
  deg        A2 -> 0     :      phi * X2
  deg        A4 -> 0     :      phi * X4

Modes: inactive (catalyst scarce, X4 <~ 1 at division) vs active
(X4 ~ 10-20 at division). Registered mode classifier at division:
active iff X4 >= 5.
"""

from __future__ import annotations

import numpy as np

KF = 1.0
KR = 1.0
PHI = 20.0
ALPHA = 100.0
KAPPA = 2400.0
VC = 1000.0

# state-change vectors for (X1, X2, X4)
DELTAS = np.array([
    [+1, 0, 0],    # transport
    [-2, +1, 0],   # R1 fwd
    [+2, -1, 0],   # R1 rev
    [0, -2, +1],   # R2 fwd
    [0, +2, -1],   # R2 rev
    [0, -1, 0],    # deg A2
    [0, 0, -1],    # deg A4
], dtype=np.int64)


def volume(X):
    return float(X[0] + 2 * X[1] + 4 * X[2])


def propensities(X, kappa=KAPPA):
    X1, X2, X4 = float(X[0]), float(X[1]), float(X[2])
    V = X1 + 2 * X2 + 4 * X4
    if V <= 0:
        return np.zeros(7)
    cat = 1.0 + kappa * X4 / V
    return np.array([
        ALPHA * X2,
        KF * X1 * max(X1 - 1, 0.0) / V * cat,
        KR * X2 * cat,
        KF * X2 * max(X2 - 1, 0.0) / V * cat,
        KR * X4 * cat,
        PHI * X2,
        PHI * X4,
    ])


def gillespie_to_division(X, rng, kappa=KAPPA, vc=VC, max_events=2_000_000):
    """Advance one protocell until V >= vc. Returns (X_at_division,
    n_events, elapsed_time, died)."""
    X = X.astype(np.int64).copy()
    t = 0.0
    for ev in range(max_events):
        if volume(X) >= vc:
            return X, ev, t, False
        a = propensities(X, kappa)
        s = a.sum()
        if s <= 0:
            return X, ev, t, True
        t += rng.exponential(1.0 / s)
        r = int(np.searchsorted(np.cumsum(a), rng.random() * s))
        X = X + DELTAS[min(r, 6)]
        if (X < 0).any():
            X = np.maximum(X, 0)
    return X, max_events, t, False


def divide(X, rng):
    """Binomial(0.5) partitioning; returns the tracked daughter
    (uniformly chosen)."""
    a = rng.binomial(X, 0.5)
    b = X - a
    return a if rng.random() < 0.5 else b


def mode_at_division(X) -> int:
    """Registered classifier: 1 = active (X4 >= 5), 0 = inactive."""
    return int(X[2] >= 5)


def run_lineage(X0, n_divisions, rng, kappa=KAPPA, vc=VC):
    """Track one lineage for n_divisions. Returns dict with per-division
    pre-division states, modes, interdivision times, and daughter
    states."""
    X = np.asarray(X0, dtype=np.int64).copy()
    pre, modes, taus, daughters = [], [], [], []
    died = False
    for _ in range(n_divisions):
        Xd, ev, tau, dead = gillespie_to_division(X, rng, kappa, vc)
        if dead:
            died = True
            break
        pre.append(Xd.copy())
        modes.append(mode_at_division(Xd))
        taus.append(tau)
        X = divide(Xd, rng)
        daughters.append(X.copy())
        if X.sum() == 0:
            died = True
            break
    return {"pre": np.array(pre), "modes": np.array(modes, dtype=int),
            "taus": np.array(taus), "daughters": np.array(daughters),
            "died": died}


INACTIVE_INIT = np.array([900, 50, 0], dtype=np.int64)    # V = 1000-
ACTIVE_INIT = np.array([850, 35, 20], dtype=np.int64)


def concentrations(X):
    V = volume(X)
    return np.array([X[0] / V, X[1] / V, X[2] / V]) if V > 0 else np.zeros(3)
