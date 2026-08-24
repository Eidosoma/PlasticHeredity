"""C7: gain/loss-of-function experiment. After every fission, exhaustively
evaluate all single-molecule additions/deletions (2*Ng candidates) and apply
the one that maximizes (max-phi) or minimizes (min-phi) Phi_r; compare
self-replicator properties vs the unintervened control runs.

Candidate scoring (paper underspecified): local Phi_r of the hypothetical
transition (current state -> candidate state) under a Gaussian model fitted to
the recent history window, with a Fiedler-vector MIB of that window.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from gard import GardParams, simulate
from composomes import label_self_replication
from phi import clr, shrunk_cov

ROOT = Path(__file__).parent.parent
WINDOW = 100 if "coarse" in sys.argv[1:] else 200
N_RUNS = 100


def _fiedler_bipartition(z: np.ndarray):
    corr = np.abs(np.nan_to_num(np.corrcoef(z, rowvar=False)))
    np.fill_diagonal(corr, 0)
    lap = np.diag(corr.sum(axis=1)) - corr
    _, vecs = np.linalg.eigh(lap)
    f = vecs[:, 1]
    m1, m2 = np.where(f >= 0)[0], np.where(f < 0)[0]
    if len(m1) == 0 or len(m2) == 0:
        half = z.shape[1] // 2
        m1, m2 = np.arange(half), np.arange(half, z.shape[1])
    return m1, m2


def _gauss_logpdf(x: np.ndarray, cov: np.ndarray, idx) -> np.ndarray:
    """Row-wise log N(x[:, idx]; 0, cov[idx, idx])."""
    i = np.asarray(idx)
    sub = cov[np.ix_(i, i)]
    chol = np.linalg.cholesky(sub)
    sol = np.linalg.solve(chol, x[:, i].T)
    quad = (sol ** 2).sum(axis=0)
    logdet = 2 * np.log(np.diag(chol)).sum()
    return -0.5 * (len(i) * np.log(2 * np.pi) + logdet + quad)


def make_intervention(sign: float, p: GardParams):
    """sign=+1 -> maximize Phi_r; sign=-1 -> minimize."""

    def intervene(counts, history, beta, params, rng, gen):
        if len(history) < WINDOW:
            return counts
        win = np.array(history[-WINDOW:])
        z = clr(win)
        sd = z.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        mu = z.mean(axis=0)
        zn = (z - mu) / sd
        d = zn.shape[1]
        m1, m2 = _fiedler_bipartition(zn)
        joint = np.hstack([zn[:-1], zn[1:]])
        cov = shrunk_cov(joint, shrink=0.05)
        past, future = np.arange(d), np.arange(d, 2 * d)

        # candidate states: add one molecule of each type, delete one of each present
        cands, edits = [], []
        for i in range(params.n_types):
            c = counts.copy(); c[i] += 1
            cands.append(c); edits.append(("add", i))
            if counts[i] > 0:
                c = counts.copy(); c[i] -= 1
                cands.append(c); edits.append(("del", i))
        zc = (clr(np.array(cands)) - mu) / sd
        z_now = np.tile(zn[-1], (len(cands), 1))
        x = np.hstack([z_now, zc])

        i_whole = (_gauss_logpdf(x, cov, past) + _gauss_logpdf(x, cov, future)
                   - _gauss_logpdf(x, cov, np.arange(2 * d)))
        phi = i_whole.copy()
        for m in (m1, m2):
            mi = (_gauss_logpdf(x, cov, m) + _gauss_logpdf(x, cov, future)
                  - _gauss_logpdf(x, cov, np.concatenate([m, future])))
            phi -= mi
        best = int(np.argmax(sign * phi))
        return cands[best]

    return intervene


def _downsample_bool(x: np.ndarray, n_out: int = 1000) -> np.ndarray:
    idx = np.linspace(0, len(x) - 1, min(n_out, len(x))).astype(int)
    return x[idx]


def _lag1_pearson(x: np.ndarray) -> float:
    x = x.astype(float)
    if x[:-1].std() < 1e-12 or x[1:].std() < 1e-12:
        return np.nan
    return float(stats.pearsonr(x[:-1], x[1:])[0])


def replicator_metrics(counts, fission_steps, seed=0):
    """Table-1 analogs. Consistency = lag-1 Pearson autocorrelation of the
    binary SR trajectory (episode compactness); *_1k metrics are computed on
    the trajectory downsampled to ~1000 samples for comparability with the
    paper's coarser molecular steps."""
    sr = label_self_replication(counts, fission_steps, seed=seed)
    n = len(sr)
    sr1k = _downsample_bool(sr)
    episodes = np.diff(np.flatnonzero(np.diff(np.r_[0, sr.view(np.int8), 0])))[::2]
    first = int(np.argmax(sr)) if sr.any() else n
    return {
        "persistence": int(sr.sum()),
        "persistence_1k": int(sr1k.sum()),
        "episode_mean": float(episodes.mean()) if len(episodes) else 0.0,
        "probability": float(sr.mean()),
        "consistency": _lag1_pearson(sr),
        "consistency_1k": _lag1_pearson(sr1k),
        "time_to_first_pct": 100.0 * first / n,
        "sr": sr,
    }


COARSE = "coarse" in sys.argv[1:]


def run_treatment(name: str, sign, outdir: Path):
    from phi import phi_r_local
    rows = []
    t0 = time.time()
    for seed in range(N_RUNS):
        p = (GardParams(dt=0.4, max_events_per_step=24.0) if COARSE
             else GardParams())
        interv = make_intervention(sign, p) if sign else None
        traj = simulate(seed=seed, p=p, intervention=interv)
        m = replicator_metrics(traj.counts, traj.fission_steps, seed=seed)
        phi, info = phi_r_local(traj.counts, mib_seed=seed)
        m["phi_mean"] = float(phi.mean())     # manipulation check
        gen = traj.generation
        sr = m.pop("sr")
        # per-generation SR probability for the trend test (C7, Fig 6C analog)
        gen_prob = [float(sr[gen == g].mean()) for g in range(p.n_gen)]
        m["gen_prob"] = gen_prob
        m["seed"] = seed
        rows.append(m)
        if (seed + 1) % 20 == 0:
            print(f"  {name}: {seed + 1}/{N_RUNS} ({time.time() - t0:.0f}s)",
                  flush=True)
    tag = "_coarse" if COARSE else ""
    (outdir / f"interv_{name}{tag}.json").write_text(json.dumps(rows))
    return rows


def summarize(all_rows: dict):
    out = {}
    for name, rows in all_rows.items():
        def col(key):
            return np.array([r[key] for r in rows], float)
        gp = np.nanmean(np.array([r["gen_prob"] for r in rows], float), axis=0)
        trend = stats.linregress(np.arange(len(gp)), gp)
        out[name] = {
            "persistence": f"{col('persistence').mean():.0f}±{col('persistence').std():.0f}",
            "persistence_1k": f"{col('persistence_1k').mean():.0f}±{col('persistence_1k').std():.0f}",
            "episode_mean": f"{col('episode_mean').mean():.0f}±{col('episode_mean').std():.0f}",
            "probability": f"{100 * col('probability').mean():.0f}±{100 * col('probability').std():.0f}%",
            "consistency_1k": f"{np.nanmean(col('consistency_1k')):.2f}±{np.nanstd(col('consistency_1k')):.2f}",
            "time_to_first": f"{col('time_to_first_pct').mean():.0f}±{col('time_to_first_pct').std():.0f}%",
            "phi_mean": f"{col('phi_mean').mean():.3f}±{col('phi_mean').std():.3f}",
            "gen_trend_slope": trend.slope, "gen_trend_p": trend.pvalue,
        }
    for metric in ("persistence", "probability", "consistency_1k", "phi_mean"):
        vals = {n: np.array([r[metric] for r in rows], float)
                for n, rows in all_rows.items()}
        out[f"mw_{metric}"] = {
            "max_vs_control": float(stats.mannwhitneyu(
                vals["max"], vals["control"], alternative="greater",
                nan_policy="omit").pvalue),
            "min_vs_control": float(stats.mannwhitneyu(
                vals["min"], vals["control"], alternative="less",
                nan_policy="omit").pvalue),
        }
    return out


def main():
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    all_rows = {}
    for name, sign in (("control", None), ("max", +1.0), ("min", -1.0)):
        print(f"treatment: {name}", flush=True)
        all_rows[name] = run_treatment(name, sign, outdir)
    summary = summarize(all_rows)
    tag = "_coarse" if COARSE else ""
    (outdir / f"interv_summary{tag}.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
