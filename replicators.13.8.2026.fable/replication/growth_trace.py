"""Phase F infrastructure: traced growth, flux vectors, and the
Kahana composition-flux alignment.

NEW code paths only — the frozen `sim._grow_gillespie` /
`sim._grow_poisson` / `sim.run_fissions` are untouched (their replay
gates continue to protect all earlier campaigns). The traced variants
reproduce the identical kinetics but (a) accept a parameterized
`nmax` (the Kahana configuration uses splitsize=1.0 -> nmax=100,
nmin=50; our frozen candidates use 80/40) and (b) snapshot the
composition at a fixed mass grid during growth.

Flux definition (verbatim from Kahana's modified tgs_grow_v10.m):
    bn        = 1 + (beta @ n) / N
    flux_i    = (Kf*rho_i*N)*bn_i - (Kb*n_i)*bn_i        (net rate)
Alignment (their `H_solver_flux`): plain cosine between composition
and flux, WITHOUT clipping to [0, 1].
"""

from __future__ import annotations

import numpy as np

import sim
from sim import NG, KF, KB, RHO, EVENTS_PER_STEP, MAXSTEPS


def flux(n: np.ndarray, beta: np.ndarray) -> np.ndarray:
    total = max(int(n.sum()), 1)
    bn = 1.0 + (beta @ n) / total
    return (KF * RHO * total) * bn - (KB * n) * bn


def cosine_signed(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def r_q(n: np.ndarray, beta: np.ndarray) -> float:
    """Kahana composition-flux alignment (unclipped cosine)."""
    return cosine_signed(n.astype(float), flux(n, beta))


def _grid_points(nmin: int, nmax: int, step: int):
    return list(range(nmin, nmax + 1, step))


def traced_grow_gillespie(n, beta, rng, nmax, grid_step=5):
    """Candidate-02 kinetics with parameterized nmax and mass-grid
    snapshots. Returns (n_final, snapshots, events, died); snapshots is
    a list of (mass, composition-copy) at first crossing of each grid
    mass (including the pre-split composition at nmax)."""
    n = n.copy()
    c = beta @ n
    total = int(n.sum())
    grid = set(_grid_points(total, nmax, grid_step))
    snaps = []
    if total in grid:
        snaps.append((total, n.copy()))
        grid.discard(total)
    events = 0
    while total < nmax:
        join, leave = sim.event_rates(n, c, total)
        rates = np.concatenate([join, leave])
        mu = sim._sample_categorical(rates, rng)
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
                return n, snaps, events, True
        events += 1
        if total in grid:
            snaps.append((total, n.copy()))
            grid.discard(total)
        if events >= 40 * MAXSTEPS:
            break
    if not snaps or snaps[-1][0] != total:
        snaps.append((total, n.copy()))
    return n, snaps, events, False


def traced_grow_poisson(n, beta, rng, nmax, grid_step=5):
    """Candidate-03 kinetics with parameterized nmax and snapshots at
    grid crossings (post-step)."""
    n = n.copy()
    total = int(n.sum())
    next_grid = ((total // grid_step) + 1) * grid_step
    snaps = [(total, n.copy())]
    steps = 0
    while total < nmax and steps < MAXSTEPS:
        c = beta @ n
        join, leave = sim.event_rates(n, c, total)
        s = join.sum() + leave.sum()
        dt = EVENTS_PER_STEP / s
        joins = rng.poisson(join * dt)
        leaves = np.minimum(rng.poisson(leave * dt), n)
        n = n + joins - leaves
        total = int(n.sum())
        steps += 1
        if total == 0:
            return n, snaps, steps, True
        if total >= next_grid:
            snaps.append((total, n.copy()))     # actual mass at crossing
            while next_grid <= total:
                next_grid += grid_step
    if snaps[-1][0] != total:
        snaps.append((total, n.copy()))
    return n, snaps, steps, False


def traced_run_fissions(n0, beta, candidate, n_fissions, rng, nmax,
                        grid_step=5, daughter_rule=None):
    """Growth+fission cycles with within-growth snapshots.

    daughter_rule defaults to the candidate's frozen semantics
    (02: equal hypergeometric split, first daughter; 03: binomial 0.5,
    uniform daughter). Returns per fission: parent, daughter, H,
    inherited, snapshots (list of (mass, comp)), events.
    """
    n = n0.copy()
    recs = []
    died = False
    for _ in range(n_fissions):
        if candidate == "02":
            n, snaps, ev, dead = traced_grow_gillespie(
                n, beta, rng, nmax, grid_step)
        else:
            n, snaps, ev, dead = traced_grow_poisson(
                n, beta, rng, nmax, grid_step)
        if dead or n.sum() < 2:
            died = True
            break
        parent = n.copy()
        if candidate == "02":
            child_a, child_b = sim._split_equal(parent, rng)
            daughter = child_a
        else:
            child_a, child_b = sim._split_binomial(parent, rng)
            pick = child_a if rng.random() < 0.5 else child_b
            if pick.sum() == 0:
                pick = child_a if child_a.sum() > 0 else child_b
            daughter = pick
        h = sim.cosine_h(parent.astype(float), daughter.astype(float))
        recs.append({"parent": parent, "daughter": daughter, "H": h,
                     "inherited": h > sim.H_THRESH, "snaps": snaps,
                     "events": ev})
        n = daughter
    return {"recs": recs, "died": died, "final": n}
