"""Original GARD (Graded Autocatalysis Replication Domain) simulator.

Implements Segré, Ben-Eli & Lancet (PNAS 2000) as specified in the Materials &
Methods of Pigozzi & Levin (arXiv 2607.28250).

Rate law (PNAS 2000, Eq. 4), split into its two nonnegative channels:
    join_i  = k_f * rho_i * N * (1 + (1/N) * sum_j beta[i,j] * n_j)
    leave_i = k_b * n_i       * (1 + (1/N) * sum_j beta[i,j] * n_j)
with k_f = 1e-2, k_b = 1e-5 s^-1, buffered environment rho_i = 1/N_g,
ln(beta_ij) ~ Normal(A, sigma^2) i.i.d. (A = -4, sigma = 4).

Stochastic scheme (PNAS 2000, Fig. 1 legend): per molecular step, each
species' join/leave count is Poisson with mean rate*dt, dt = 0.05 s; a step
whose expected total event count exceeds `max_events_per_step` is subdivided
(halved dt) to bound overshoot when catalytic rates spike.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class GardParams:
    n_types: int = 100          # N_g
    n_min: int = 40
    n_max: int = 80             # = 2 * n_min, fission trigger
    n_gen: int = 100
    max_steps: int = 1000       # per generation
    mu_beta: float = -4.0       # A: mean of ln(beta)
    sigma_beta: float = 4.0     # sd of ln(beta)
    k_f: float = 1e-2
    k_b: float = 1e-5
    dt: float = 0.05
    max_events_per_step: float = 16.0

    @property
    def rho(self) -> float:
        return 1.0 / self.n_types


@dataclass
class GardTrajectory:
    counts: np.ndarray            # (n_steps, n_types) composition after each step
    generation: np.ndarray        # (n_steps,) generation index of each step
    fission_steps: np.ndarray     # step indices at which fission occurred
    joins: np.ndarray             # (n_steps, n_types)
    leaves: np.ndarray            # (n_steps, n_types)
    params: GardParams = field(repr=False, default=None)


def sample_beta(rng: np.random.Generator, p: GardParams) -> np.ndarray:
    return rng.lognormal(mean=p.mu_beta, sigma=p.sigma_beta,
                         size=(p.n_types, p.n_types))


def initial_assembly(rng: np.random.Generator, p: GardParams) -> np.ndarray:
    counts = np.zeros(p.n_types, dtype=np.int64)
    types = rng.choice(p.n_types, size=p.n_min, replace=False)
    counts[types] = 1
    return counts


def rates(counts: np.ndarray, p: GardParams, beta: np.ndarray):
    n_total = counts.sum()
    catalysis = 1.0 + (beta @ counts) / max(n_total, 1)
    lam_f = p.k_f * p.rho * n_total * catalysis
    lam_b = p.k_b * counts * catalysis
    return lam_f, lam_b


def step(counts: np.ndarray, beta: np.ndarray, p: GardParams,
         rng: np.random.Generator):
    """One molecular step: Poisson draws of joins/leaves per species over dt,
    with dt halved as needed so expected events stay below the cap."""
    lam_f, lam_b = rates(counts, p, beta)
    dt = p.dt
    lam_tot = lam_f.sum() + lam_b.sum()
    while lam_tot * dt > p.max_events_per_step:
        dt *= 0.5
    joins = rng.poisson(lam_f * dt)
    leaves = np.minimum(rng.poisson(lam_b * dt), counts)
    return counts + joins - leaves, joins, leaves


def fission(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(counts, 0.5)


def simulate(seed: int, p: GardParams = None, beta: np.ndarray = None,
             intervention=None) -> GardTrajectory:
    """Run one GARD assembly for p.n_gen growth-fission generations.

    intervention: optional callable (counts, history, beta, p, rng, gen) -> counts,
    applied to the surviving daughter immediately after each fission (used for
    the max/min-Phi experiments); history is the list of per-step count arrays.
    """
    p = p or GardParams()
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = sample_beta(rng, p)
    counts = initial_assembly(rng, p)

    all_counts, all_joins, all_leaves, gen_idx, fission_steps = [], [], [], [], []
    step_i = 0
    for gen in range(p.n_gen):
        for _ in range(p.max_steps):
            counts, joins, leaves = step(counts, beta, p, rng)
            all_counts.append(counts.copy())
            all_joins.append(joins)
            all_leaves.append(leaves)
            gen_idx.append(gen)
            step_i += 1
            if counts.sum() >= p.n_max:
                break
        fission_steps.append(step_i - 1)
        counts = fission(counts, rng)
        if counts.sum() == 0:
            counts = initial_assembly(rng, p)
        if intervention is not None:
            counts = intervention(counts, all_counts, beta, p, rng, gen)

    return GardTrajectory(
        counts=np.array(all_counts),
        generation=np.array(gen_idx),
        fission_steps=np.array(fission_steps),
        joins=np.array(all_joins),
        leaves=np.array(all_leaves),
        params=p,
    )
