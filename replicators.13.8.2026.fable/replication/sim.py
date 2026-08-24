"""GARD simulator: two frozen candidate contracts.

Kinetics pinned to historical GARD10 source (ModelingOriginsofLife/GARD,
commit 86dff6320d5ae91b4e831471079ff46749b14df9, tgs_grow_v10.m):

    bn_i   = 1 + (1/N) * (beta @ n)_i
    join_i = Kf * rho_i * N * bn_i          (rho uniform = 1/NG)
    leave_i = Kb * n_i * bn_i

Parameters follow the target paper's Materials & Methods (which follow
Segre et al. 2000 / GARD10 defaults): NG=100, nmin=40, nmax=80,
beta_ij = exp(A + sigma*Z), A=-4, sigma=4, Kf=1e-2, Kb=1e-4, ngen=100,
maxsteps=1000.

Candidate contracts (exposure / fission / daughter / trim semantics):

  candidate 02 ("historical"):
    - direct categorical (Gillespie-like) single-molecule events
    - growth terminates exactly at N == nmax (no overshoot)
    - fission: equal stochastic split (sequential weighted sampling
      without replacement, as tgs_split_v10) -> both daughters mass 40
    - the FIRST daughter continues the lineage

  candidate 03 ("paper-described vector-Poisson"):
    - vector-Poisson exposure: per step, Poisson joins with mean
      join_i * dt and Poisson leaves with mean leave_i * dt (clipped at
      n_i), dt chosen adaptively so the expected total event count per
      step is EVENTS_PER_STEP
    - overshoot allowed: growth stops at N >= nmax, no trim
    - fission: independent binomial(0.5) per molecule
    - a UNIFORMLY selected daughter continues the lineage
"""

from __future__ import annotations

import numpy as np

NG = 100
NMIN = 40
NMAX = 80
A_MU = -4.0
SIGMA = 4.0
KF = 1e-2
KB = 1e-4
RHO = 1.0 / NG          # uniform environmental concentration per type
MAXSTEPS = 1000
NGEN = 100
EVENTS_PER_STEP = 4.0   # candidate 03 vector-Poisson exposure scale
H_THRESH = 0.9          # strict inheritance threshold (cosine)


def make_beta(rng: np.random.Generator, a_mu: float = A_MU,
              sigma: float = SIGMA) -> np.ndarray:
    """Lognormal catalytic matrix. Defaults reproduce the frozen
    campaigns exactly; a_mu/sigma overrides exist only for the
    parameter-regime probe."""
    return np.exp(rng.standard_normal((NG, NG)) * sigma + a_mu)


def make_initial_state(rng: np.random.Generator) -> np.ndarray:
    """Mass-40 distinct-singleton initial assembly (types sampled
    uniformly without replacement, per target paper)."""
    n = np.zeros(NG, dtype=np.int64)
    types = rng.choice(NG, size=NMIN, replace=False)
    n[types] = 1
    return n


def event_rates(n: np.ndarray, c: np.ndarray, total: int):
    """Join/leave propensities from the boost vector c = beta @ n.

    Pure function of the historical GARD equation (tgs_grow_v10.m):
        bn      = 1 + c / N
        join_i  = Kf * rho_i * N * bn_i
        leave_i = Kb * n_i * bn_i
    """
    bn = 1.0 + c / total
    join = (KF * RHO * total) * bn
    leave = (KB * n) * bn
    return join, leave


def _sample_categorical(rates: np.ndarray, rng: np.random.Generator) -> int:
    """Draw one event index proportional to `rates` (cumsum/searchsorted,
    identical to the original inline logic — RNG call order preserved)."""
    s = rates.sum()
    u = rng.random() * s
    cs = np.cumsum(rates)
    mu = int(np.searchsorted(cs, u))
    if mu >= len(rates):
        mu = len(rates) - 1
    return mu


def cosine_h(x: np.ndarray, y: np.ndarray) -> float:
    nx = float(np.sqrt(np.dot(x, x)))
    ny = float(np.sqrt(np.dot(y, y)))
    if nx < 1e-7 or ny < 1e-7:
        return 0.0
    h = float(np.dot(x, y) / (nx * ny))
    return min(1.0, max(0.0, h))


def _grow_gillespie(n: np.ndarray, beta: np.ndarray, rng: np.random.Generator):
    """Candidate 02 growth: single-molecule categorical events until
    N == NMAX exactly. Maintains c = beta @ n incrementally."""
    n = n.copy()
    c = beta @ n            # (NG,)
    total = int(n.sum())
    events = 0
    while total < NMAX:
        join, leave = event_rates(n, c, total)
        rates = np.concatenate([join, leave])
        mu = _sample_categorical(rates, rng)
        if mu < NG:
            n[mu] += 1
            c += beta[:, mu]
            total += 1
        else:
            k = mu - NG
            n[k] -= 1
            c -= beta[:, k]
            total -= 1
            if total == 0:
                return n, events, True
        events += 1
        if events >= 40 * MAXSTEPS:   # safety valve, never hit in practice
            break
    return n, events, False


def _grow_poisson(n: np.ndarray, beta: np.ndarray, rng: np.random.Generator):
    """Candidate 03 growth: vector-Poisson exposure steps until
    N >= NMAX (overshoot allowed) or MAXSTEPS."""
    n = n.copy()
    total = int(n.sum())
    steps = 0
    while total < NMAX and steps < MAXSTEPS:
        c = beta @ n
        join, leave = event_rates(n, c, total)
        s = join.sum() + leave.sum()
        dt = EVENTS_PER_STEP / s
        joins = rng.poisson(join * dt)
        leaves = np.minimum(rng.poisson(leave * dt), n)
        n = n + joins - leaves
        total = int(n.sum())
        steps += 1
        if total == 0:
            return n, steps, True
    return n, steps, False


def _split_equal(parent: np.ndarray, rng: np.random.Generator):
    """Candidate 02 fission: sequential weighted draws without
    replacement into child A until it holds floor(N/2) molecules
    (multivariate hypergeometric); child B is the remainder."""
    total = int(parent.sum())
    child_a = rng.multivariate_hypergeometric(parent, total // 2)
    child_b = parent - child_a
    return child_a.astype(np.int64), child_b.astype(np.int64)


def _split_binomial(parent: np.ndarray, rng: np.random.Generator):
    """Candidate 03 fission: each molecule independently binomial(0.5)."""
    child_a = rng.binomial(parent, 0.5)
    child_b = parent - child_a
    return child_a.astype(np.int64), child_b.astype(np.int64)


def run_fissions(
    n0: np.ndarray,
    beta: np.ndarray,
    candidate: str,
    n_fissions: int,
    rng: np.random.Generator,
):
    """Run `n_fissions` growth+fission cycles from post-fission state n0.

    Returns dict with per-fission records:
      parents  : (F, NG) pre-fission compositions
      daughters: (F, NG) selected post-fission compositions
      H        : (F,) strict parent->selected-daughter cosine similarity
      inherited: (F,) bool, H > 0.9
      died     : bool
      n_done   : number of completed fissions
    """
    n = n0.copy()
    parents = np.zeros((n_fissions, NG), dtype=np.int64)
    daughters = np.zeros((n_fissions, NG), dtype=np.int64)
    hs = np.zeros(n_fissions)
    updates = np.zeros(n_fissions, dtype=np.int64)
    died = False
    f_done = 0
    for f in range(n_fissions):
        if candidate == "02":
            n, n_upd, dead = _grow_gillespie(n, beta, rng)
        elif candidate == "03":
            n, n_upd, dead = _grow_poisson(n, beta, rng)
        else:
            raise ValueError(candidate)
        updates[f] = n_upd
        if dead or n.sum() < 2:
            died = True
            break
        parent = n.copy()
        if candidate == "02":
            child_a, child_b = _split_equal(parent, rng)
            daughter = child_a                      # first daughter continues
        else:
            child_a, child_b = _split_binomial(parent, rng)
            pick = child_a if rng.random() < 0.5 else child_b
            if pick.sum() == 0:                     # degenerate empty daughter
                pick = child_a if child_a.sum() > 0 else child_b
            daughter = pick                         # uniform daughter continues
        parents[f] = parent
        daughters[f] = daughter
        hs[f] = cosine_h(parent.astype(float), daughter.astype(float))
        n = daughter
        f_done += 1
    return {
        "parents": parents[:f_done],
        "daughters": daughters[:f_done],
        "H": hs[:f_done],
        "inherited": hs[:f_done] > H_THRESH,
        "updates": updates[:f_done],   # events (cand 02) / steps (cand 03)
        "died": died,
        "n_done": f_done,
        "final": n,
    }
