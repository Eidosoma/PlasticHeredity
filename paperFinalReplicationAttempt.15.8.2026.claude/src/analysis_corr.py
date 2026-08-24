"""Analyses C1-C4 and C6: aggregate trend, spikes, Phi-vs-self-replication
correlations, temporal structure, spike-timing correlations.

Reads results/runs/run_*.npz; writes results/corr_stats.json and figures/.
Alignment: phi[t] (transition t->t+1) is paired with sr[t+1].
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

np.seterr(all="ignore")
ROOT = Path(__file__).parent.parent
LAG = 1


RUNS_SUB = "runs_coarse" if "coarse" in sys.argv[1:] else "runs"
STATS_NAME = ("corr_stats_coarse.json" if "coarse" in sys.argv[1:]
              else "corr_stats.json")


def load_runs():
    runs = []
    for f in sorted((ROOT / "results" / RUNS_SUB).glob("run_*.npz")):
        d = np.load(f)
        runs.append({
            "phi": d["phi"].astype(float),
            "sr": d["sr"][LAG:],           # aligned to phi
            "sr_full": d["sr"],
            "seed": int(f.stem.split("_")[1]),
        })
    return runs


def spike_mask(phi):
    return phi > phi.mean() + 3 * phi.std()


def analyze(runs):
    out = {}

    # C1: aggregate trend over molecular steps (cross-run median where >=50% alive)
    max_len = max(len(r["phi"]) for r in runs)
    counts_alive = np.zeros(max_len, dtype=int)
    for r in runs:
        counts_alive[:len(r["phi"])] += 1
    horizon = int(np.max(np.where(counts_alive >= len(runs) // 2)[0])) + 1
    agg = np.full((len(runs), horizon), np.nan)
    for i, r in enumerate(runs):
        n = min(len(r["phi"]), horizon)
        agg[i, :n] = r["phi"][:n]
    med = np.nanmedian(agg, axis=0)
    sd = np.nanstd(agg, axis=0)
    x = np.arange(horizon)
    lr = stats.linregress(x, med)
    out["C1_aggregate_trend"] = {"slope": lr.slope, "p": lr.pvalue,
                                 "horizon_steps": horizon}

    # C2: punctuated spikes
    n_spike_runs = sum(spike_mask(r["phi"]).any() for r in runs)
    out["C2_spikes"] = {"runs_with_spikes": n_spike_runs, "n_runs": len(runs)}

    # C3: per-run Spearman phi vs sr; Mann-Whitney phi | sr
    rhos, rho_ps, mw_ps, higher = [], [], [], []
    for r in runs:
        if r["sr"].min() == r["sr"].max():
            rhos.append(np.nan); rho_ps.append(np.nan); mw_ps.append(np.nan)
            higher.append(False)
            continue
        rho, p = stats.spearmanr(r["phi"], r["sr"].astype(float))
        rhos.append(rho); rho_ps.append(p)
        a, b = r["phi"][r["sr"]], r["phi"][~r["sr"]]
        mw = stats.mannwhitneyu(a, b, alternative="greater")
        mw_ps.append(mw.pvalue)
        higher.append(np.median(a) > np.median(b))
    rhos, rho_ps, mw_ps = map(np.array, (rhos, rho_ps, mw_ps))
    valid = ~np.isnan(rhos)
    pos = (rhos > 0) & valid
    pos_sig = pos & (rho_ps < 0.05)
    hi_sig = np.array(higher) & (mw_ps < 0.001)
    fisher_stat = -2 * np.nansum(np.log(np.clip(mw_ps[valid], 1e-300, 1)))
    fisher_p = stats.chi2.sf(fisher_stat, 2 * valid.sum())
    tt = stats.ttest_1samp(rhos[valid], 0)
    out["C3_correlation"] = {
        "runs_valid": int(valid.sum()),
        "positive": int(pos.sum()), "positive_significant": int(pos_sig.sum()),
        "phi_higher_in_sr_p001": int(hi_sig.sum()),
        "fisher_p": float(fisher_p),
        "mean_rho": float(np.nanmean(rhos)), "mean_rho_ttest_p": float(tt.pvalue),
    }

    # C4: Ljung-Box temporal structure (subsample long series for tractability)
    lb_reject, lb_reject_diff, lb_ps = 0, 0, []
    for r in runs:
        phi = r["phi"]
        p_lb = acorr_ljungbox(phi, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
        lb_ps.append(p_lb)
        if p_lb < 0.05:
            lb_reject += 1
        p_lbd = acorr_ljungbox(np.diff(phi), lags=[20],
                               return_df=True)["lb_pvalue"].iloc[0]
        if p_lbd < 0.05:
            lb_reject_diff += 1
    out["C4_ljungbox"] = {"reject": lb_reject, "reject_after_diff": lb_reject_diff,
                          "median_p": float(np.median(lb_ps))}

    # C6: spike timing/distance/height vs self-replication probability (across runs)
    sr_prob, mean_t, mean_gap, mean_h = [], [], [], []
    for r in runs:
        m = spike_mask(r["phi"])
        if not m.any():
            continue
        t = np.where(m)[0]
        sr_prob.append(r["sr"].mean())
        mean_t.append(t.mean() / len(r["phi"]))
        mean_gap.append(np.diff(t).mean() / len(r["phi"]) if len(t) > 1 else np.nan)
        mean_h.append((r["phi"][t] - r["phi"].mean()).mean() / r["phi"].std())
    def sp(a, b):
        a, b = np.array(a), np.array(b)
        ok = ~(np.isnan(a) | np.isnan(b))
        rho, p = stats.spearmanr(a[ok], b[ok])
        return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}
    out["C6_spike_geometry"] = {
        "sr_prob_vs_spike_time": sp(sr_prob, mean_t),
        "sr_prob_vs_spike_gap": sp(sr_prob, mean_gap),
        "sr_prob_vs_spike_height": sp(sr_prob, mean_h),
    }

    return out, (x, med, sd), rhos


def figures(runs, aggregate, rhos):
    figdir = ROOT / "figures"
    figdir.mkdir(exist_ok=True)
    x, med, sd = aggregate

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(x, med, lw=0.8)
    ax.fill_between(x, med - sd, med + sd, alpha=0.2)
    lr = stats.linregress(x, med)
    ax.plot(x, lr.intercept + lr.slope * x, "r", lw=1,
            label=f"p={lr.pvalue:.3f}")
    ax.legend(); ax.set_title("Aggregate $\\Phi_r$ (median$\\pm$std)")
    ax.set_xlabel("molecular step"); ax.set_ylabel("$\\Phi_r$")
    for ax, r in zip(axes.flat[1:], runs[:3]):
        ax.plot(r["phi"], lw=0.4)
        thr = r["phi"].mean() + 3 * r["phi"].std()
        ax.axhline(thr, color="r", ls="--", lw=0.8)
        ax.set_title(f"run {r['seed']}")
        ax.set_xlabel("molecular step"); ax.set_ylabel("$\\Phi_r$")
    fig.tight_layout(); fig.savefig(figdir / "fig2_trajectories.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    v = rhos[~np.isnan(rhos)]
    axes[0].hist(v, bins=25)
    axes[0].axvline(0, color="b", ls="--"); axes[0].axvline(v.mean(), color="r", ls="--")
    axes[0].set_title("Spearman $\\rho$($\\Phi_r$, self-replication) per run")
    means_sr, means_dr = [], []
    for r in runs:
        if r["sr"].min() != r["sr"].max():
            means_dr.append(r["phi"][~r["sr"]].mean())
            means_sr.append(r["phi"][r["sr"]].mean())
    for a, b in zip(means_dr, means_sr):
        axes[1].plot([0, 1], [a, b], "k-", alpha=0.15)
    axes[1].set_xticks([0, 1], ["drift", "self-replicating"])
    axes[1].set_ylabel("mean $\\Phi_r$")
    axes[1].set_title(f"$\\Phi_r$ higher in SR: {sum(b>a for a,b in zip(means_dr,means_sr))}"
                      f"/{len(means_sr)} runs")
    fig.tight_layout(); fig.savefig(figdir / "fig34_correlation.png", dpi=150)


def main():
    runs = load_runs()
    print(f"loaded {len(runs)} runs")
    out, aggregate, rhos = analyze(runs)
    (ROOT / "results" / STATS_NAME).write_text(
        json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    if RUNS_SUB == "runs":
        figures(runs, aggregate, rhos)
        print("figures written")


if __name__ == "__main__":
    main()
