"""C7 recovery: coarse-universe interventions with rollout candidate scoring.

For each fission, every candidate (single add/delete, 2*Ng max) is scored by
the mean local Phi_r over a K-step simulated rollout from the candidate state
(common random numbers across candidates), under a Gaussian model fitted to
the trailing window. Metrics use the calibrated detector (per-run quantile,
88% prevalence) plus the classic 0.9 labels; consistency = lag-1 autocorr of
the Phi_r trajectory. Run counts are saved for post-hoc analyses.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from gard import GardParams, simulate, rates
from composomes import label_self_replication
from recovery_c5 import quantile_labels
from phi import clr, shrunk_cov, phi_r_local
from interventions import _fiedler_bipartition, _gauss_logpdf

ROOT = Path(__file__).parent.parent
WINDOW = 100
ROLLOUT = 10
N_RUNS = 100


def rollout(counts, beta, p, rng_seed, k=ROLLOUT):
    rng = np.random.default_rng(rng_seed)
    out = np.empty((k, len(counts)), dtype=float)
    c = counts.copy()
    for t in range(k):
        lam_f, lam_b = rates(c, p, beta)
        lam_tot = lam_f.sum() + lam_b.sum()
        dt = p.dt
        while lam_tot * dt > p.max_events_per_step:
            dt *= 0.5
        joins = rng.poisson(lam_f * dt)
        leaves = np.minimum(rng.poisson(lam_b * dt), c)
        c = c + joins - leaves
        out[t] = c
    return out


def make_rollout_intervention(sign: float):
    def intervene(counts, history, beta, params, rng, gen):
        if len(history) < WINDOW:
            return counts
        win = np.array(history[-WINDOW:])
        z = clr(win)
        mu, sd = z.mean(axis=0), z.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        zn = (z - mu) / sd
        d = zn.shape[1]
        m1, m2 = _fiedler_bipartition(zn)
        joint = np.hstack([zn[:-1], zn[1:]])
        cov = shrunk_cov(joint, shrink=0.05)
        past, future = np.arange(d), np.arange(d, 2 * d)
        all_idx = np.arange(2 * d)
        m1f = np.concatenate([m1, future])
        m2f = np.concatenate([m2, future])

        cands = []
        for i in range(params.n_types):
            c = counts.copy(); c[i] += 1
            cands.append(c)
            if counts[i] > 0:
                c = counts.copy(); c[i] -= 1
                cands.append(c)

        crn_seed = int(rng.integers(1 << 31))
        scores = np.empty(len(cands))
        for ci, cand in enumerate(cands):
            traj = rollout(cand, beta, params, crn_seed)
            zs = (clr(np.vstack([cand[None, :], traj])) - mu) / sd
            x = np.hstack([zs[:-1], zs[1:]])
            phi = (_gauss_logpdf(x, cov, past) + _gauss_logpdf(x, cov, future)
                   - _gauss_logpdf(x, cov, all_idx))
            phi -= (_gauss_logpdf(x, cov, m1) + _gauss_logpdf(x, cov, future)
                    - _gauss_logpdf(x, cov, m1f))
            phi -= (_gauss_logpdf(x, cov, m2) + _gauss_logpdf(x, cov, future)
                    - _gauss_logpdf(x, cov, m2f))
            scores[ci] = phi.mean()
        return cands[int(np.argmax(sign * scores))]
    return intervene


def metrics_for(counts, fission_steps, phi, seed):
    out = {}
    for rule, sr in (("classic09",
                      label_self_replication(counts, fission_steps, seed=seed)),
                     ("quantile12",
                      quantile_labels(counts, fission_steps, seed=seed))):
        first = int(np.argmax(sr)) if sr.any() else len(sr)
        out[rule] = {"persistence": int(sr.sum()),
                     "probability": float(sr.mean()),
                     "time_to_first": first}
    out["consistency_phi"] = float(stats.pearsonr(phi[:-1], phi[1:])[0])
    out["phi_mean"] = float(phi.mean())
    return out


def main():
    outdir = ROOT / "results" / "recovery_c7_runs"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, sign in (("control", None), ("max", +1.0), ("min", -1.0)):
        rows = []
        t0 = time.time()
        for seed in range(N_RUNS):
            p = GardParams(dt=0.4, max_events_per_step=24.0)
            interv = make_rollout_intervention(sign) if sign else None
            traj = simulate(seed=seed, p=p, intervention=interv)
            phi, _ = phi_r_local(traj.counts, mib_seed=seed)
            m = metrics_for(traj.counts, traj.fission_steps, phi, seed)
            m["seed"] = seed
            rows.append(m)
            np.savez_compressed(outdir / f"{name}_{seed:03d}.npz",
                                counts=traj.counts.astype(np.int16),
                                fission_steps=traj.fission_steps,
                                phi=phi.astype(np.float32))
            if (seed + 1) % 25 == 0:
                print(f"  {name}: {seed + 1}/{N_RUNS} ({time.time() - t0:.0f}s)",
                      flush=True)
        summary[name] = rows
    (ROOT / "results" / "recovery_c7_rows.json").write_text(
        json.dumps(summary, default=float))

    report = {}
    for name, rows in summary.items():
        rep = {}
        for rule in ("classic09", "quantile12"):
            for k in ("persistence", "probability", "time_to_first"):
                v = np.array([r[rule][k] for r in rows], float)
                rep[f"{rule}.{k}"] = f"{v.mean():.2f}±{v.std():.2f}"
        for k in ("consistency_phi", "phi_mean"):
            v = np.array([r[k] for r in rows], float)
            rep[k] = f"{v.mean():.3f}±{v.std():.3f}"
        report[name] = rep
    tests = {}
    for rule in ("classic09", "quantile12"):
        for k in ("persistence", "probability"):
            a = np.array([r[rule][k] for r in summary["max"]], float)
            b = np.array([r[rule][k] for r in summary["control"]], float)
            c = np.array([r[rule][k] for r in summary["min"]], float)
            tests[f"{rule}.{k}"] = {
                "max>control": float(stats.mannwhitneyu(a, b, alternative="greater").pvalue),
                "min<control": float(stats.mannwhitneyu(c, b, alternative="less").pvalue)}
    for k in ("consistency_phi", "phi_mean"):
        a = np.array([r[k] for r in summary["max"]], float)
        b = np.array([r[k] for r in summary["control"]], float)
        c = np.array([r[k] for r in summary["min"]], float)
        tests[k] = {
            "max>control": float(stats.mannwhitneyu(a, b, alternative="greater").pvalue),
            "min<control": float(stats.mannwhitneyu(c, b, alternative="less").pvalue)}
    out = {"means": report, "tests": tests}
    (ROOT / "results" / "recovery_c7.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
