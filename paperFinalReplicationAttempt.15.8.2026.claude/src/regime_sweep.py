"""Dynamics-regime sweep: find a GARD regime (consistent with the paper's
stated Ng/nmin/nmax/A/sigma) whose run fates are as predictable from the first
25% of steps as the paper's Fig. 5 implies (~80% baseline accuracy).

Grid: scheme (tau-leap coarse | Gillespie event-per-event, subsampled to ~10
steps/gen) x k_b x mu_beta x k_f*rho. 30 runs/cell. Cell scores:
  heredity  = consecutive-generation cosine H
  prev      = SR prevalence (classic 0.9 cosine to dominant compotype)
  ntot      = molecular steps per run
  fate_rho2 = cross-run Spearman rho^2 between early (first 25%) mean flux and
              late (last 75%) SR fraction  — the predictability proxy
  fate_r2   = LOO linear R^2 from 3 early features -> late SR fraction
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from gard import GardParams, sample_beta, initial_assembly, fission
from composomes import nondrift_mask, compotypes, cosine_to

ROOT = Path(__file__).parent.parent
N_RUNS = 30


def simulate_tau(seed, p, kfrho):
    rng = np.random.default_rng(seed)
    beta = sample_beta(rng, p)
    counts = initial_assembly(rng, p)
    all_counts, fission_steps, step_i = [], [], 0
    for gen in range(p.n_gen):
        for _ in range(p.max_steps):
            n_total = counts.sum()
            cat = 1.0 + (beta @ counts) / max(n_total, 1)
            lam_f = kfrho * n_total * cat
            lam_b = p.k_b * counts * cat
            lam_tot = lam_f.sum() + lam_b.sum()
            dt = p.dt
            while lam_tot * dt > p.max_events_per_step:
                dt *= 0.5
            counts = counts + rng.poisson(lam_f * dt) \
                - np.minimum(rng.poisson(lam_b * dt), counts)
            all_counts.append(counts.copy())
            step_i += 1
            if counts.sum() >= p.n_max:
                break
        fission_steps.append(step_i - 1)
        counts = fission(counts, rng)
        if counts.sum() == 0:
            counts = initial_assembly(rng, p)
    return np.array(all_counts), np.array(fission_steps)


def simulate_gillespie(seed, p, kfrho, record_every=8):
    """Event-per-event kinetic MC with incremental catalysis updates;
    compositions recorded every `record_every` events (~10 steps/gen)."""
    rng = np.random.default_rng(seed)
    beta = sample_beta(rng, p)
    counts = initial_assembly(rng, p)
    all_counts, fission_steps, step_i = [], [], 0
    for gen in range(p.n_gen):
        catsum = beta @ counts
        events = 0
        while True:
            n_total = counts.sum()
            cat = 1.0 + catsum / max(n_total, 1)
            lam_f = kfrho * n_total * cat
            lam_b = p.k_b * counts * cat
            lam = np.concatenate([lam_f, lam_b])
            tot = lam.sum()
            if tot <= 0 or events >= p.max_steps * record_every:
                break
            ev = rng.choice(2 * p.n_types, p=lam / tot)
            if ev < p.n_types:
                counts[ev] += 1
                catsum += beta[:, ev]
            else:
                t = ev - p.n_types
                counts[t] -= 1
                catsum -= beta[:, t]
            events += 1
            if events % record_every == 0:
                all_counts.append(counts.copy())
                step_i += 1
            if counts.sum() >= p.n_max:
                break
        if events % record_every != 0:
            all_counts.append(counts.copy())
            step_i += 1
        fission_steps.append(step_i - 1)
        counts = fission(counts, rng)
        if counts.sum() == 0:
            counts = initial_assembly(rng, p)
    return np.array(all_counts), np.array(fission_steps)


def sr_labels(counts, fs):
    pre = counts[fs].astype(float)
    mask = nondrift_mask(pre, 0.9)
    if mask.sum() >= 2:
        cents, labels = compotypes(pre[mask], seed=0)
        dom = cents[np.bincount(labels).argmax()]
    else:
        dom = pre[0]
    return cosine_to(counts.astype(float), dom) >= 0.9


def cell_metrics(scheme, kb, mu, kfrho):
    heredity, prev, ntot = [], [], []
    early_flux, early_sr, early_sim, late_sr = [], [], [], []
    for seed in range(N_RUNS):
        p = GardParams(dt=0.4, max_events_per_step=24.0, k_b=kb, mu_beta=mu)
        sim = simulate_gillespie if scheme == "gillespie" else simulate_tau
        counts, fs = sim(seed, p, kfrho)
        if len(counts) < 100 or len(fs) < 10:
            continue
        pre = counts[fs].astype(float)
        u = pre / np.linalg.norm(pre, axis=1, keepdims=True)
        heredity.append(float(np.median(np.sum(u[:-1] * u[1:], axis=1))))
        sr = sr_labels(counts, fs)
        prev.append(sr.mean())
        ntot.append(len(counts))
        cut = int(len(counts) * 0.25)
        flux = np.abs(np.diff(counts.astype(float), axis=0)).sum(axis=1)
        early_flux.append(flux[:cut].mean())
        early_sr.append(sr[:cut].mean())
        early_sim.append(float(np.mean(sr[:cut])))
        late_sr.append(sr[cut:].mean())
    if len(late_sr) < 10:
        return None
    rho, _ = stats.spearmanr(early_flux, late_sr)
    X = np.stack([early_flux, early_sr, early_sim], axis=1)
    y = np.array(late_sr)
    preds = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        coef, *_ = np.linalg.lstsq(
            np.c_[X[m], np.ones(m.sum())], y[m], rcond=None)
        preds.append(np.r_[X[i], 1] @ coef)
    ss = 1 - np.sum((y - preds) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return dict(heredity=float(np.median(heredity)),
                prev=float(np.mean(prev)), ntot=int(np.median(ntot)),
                fate_rho2=float(rho ** 2), fate_r2=float(ss),
                n=len(late_sr))


def main():
    grid = list(itertools.product(
        ["tau", "gillespie"], [1e-5, 1e-4, 1e-3], [-4.0, -3.0], [1e-4, 1e-3]))
    results = {}
    t0 = time.time()
    for scheme, kb, mu, kfrho in grid:
        key = f"{scheme}/kb{kb:g}/mu{mu:g}/kfrho{kfrho:g}"
        m = cell_metrics(scheme, kb, mu, kfrho)
        results[key] = m
        print(f"[{time.time()-t0:5.0f}s] {key}: {m}", flush=True)
    (ROOT / "results" / "regime_sweep.json").write_text(
        json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
