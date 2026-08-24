"""Validation against PNAS 2000 Fig. 4 benchmarks:
parent-vs-daughter compositional similarity H should be ~0.65±0.07 with no
catalysis (beta=0), ~0.81±0.10 at mu=-6, ~0.88±0.10 at mu=-4.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from gard import GardParams, simulate, sample_beta
from composomes import _unit


def parent_daughter_h(seeds, p, zero_beta=False):
    vals = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        beta = sample_beta(rng, p)
        if zero_beta:
            beta = np.zeros_like(beta)
        traj = simulate(seed=seed, p=p, beta=beta)
        pre = _unit(traj.counts[traj.fission_steps].astype(float))
        vals.extend(np.sum(pre[:-1] * pre[1:], axis=1))
    return np.mean(vals), np.std(vals)


if __name__ == "__main__":
    seeds = range(10)
    for label, mu, zero in (("beta=0 (target 0.65±0.07)", -4.0, True),
                            ("mu=-6  (target 0.81±0.10)", -6.0, False),
                            ("mu=-4  (target 0.88±0.10)", -4.0, False)):
        m, s = parent_daughter_h(seeds, GardParams(mu_beta=mu), zero_beta=zero)
        print(f"{label}: H = {m:.2f} ± {s:.2f}")
