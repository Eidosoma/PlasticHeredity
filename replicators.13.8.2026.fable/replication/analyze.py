"""Confirmation analysis (L54-analog): reliability, rank transfer,
proper scores, matrix bootstraps, whole-matrix permutations, process
probabilities, and figures."""

import json
import os
import pickle

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "results")
FIG = os.path.join(OUT, "figures")

N_BOOT = 4096
N_PERM = 512
CANDIDATES = ["02", "03"]

BLUE = "#4878A8"
INK = "#33383D"
MUTED = "#6B7178"
GRID = "#D9DDE1"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9,
})


def sp(a, b):
    return float(spearmanr(a, b).correlation)


def center_by(v, groups):
    v = np.asarray(v, dtype=float)
    _, inv = np.unique(groups, return_inverse=True)
    means = np.bincount(inv, weights=v) / np.bincount(inv)
    return v - means[inv]


def logloss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def matrix_boot(stat_fn, mats, rng, n=N_BOOT):
    """Bootstrap a statistic by resampling whole matrices."""
    u = np.unique(mats)
    idx_map = {m: np.where(mats == m)[0] for m in u}
    vals = np.empty(n)
    for i in range(n):
        pick = rng.choice(u, size=len(u), replace=True)
        idx = np.concatenate([idx_map[m] for m in pick])
        vals[i] = stat_fn(idx)
    return vals


def analyze_candidate(rows):
    mats = np.array([r["matrix"] for r in rows])
    qA = np.array([r["qA"] for r in rows])
    qB = np.array([r["qB"] for r in rows])
    q = np.array([r["q"] for r in rows])
    Y = np.stack([r["y64"] for r in rows]).astype(float)   # (S, 64)
    P = {k: np.array([r[f"p_{k}"] for r in rows])
         for k in ["full", "direct", "beta", "prior"]}
    rng = np.random.default_rng(20260813)
    res = {"n_states": len(rows)}

    res["q_transition_count"] = int(np.sum((q > 0.1) & (q < 0.9)))

    # --- branch-half reliability -------------------------------------
    res["reliability"] = sp(qA, qB)
    boot = matrix_boot(lambda i: sp(qA[i], qB[i]), mats, rng)
    res["reliability_lower95"] = float(np.quantile(boot, 0.025))
    cA, cB = center_by(qA, mats), center_by(qB, mats)
    res["reliability_centered"] = sp(cA, cB)
    boot = matrix_boot(
        lambda i: sp(center_by(qA[i], mats[i]), center_by(qB[i], mats[i])),
        mats, rng)
    res["reliability_centered_lower95"] = float(np.quantile(boot, 0.025))

    # --- rank transfer (against each branch half = "direction") ------
    for model in ["full", "direct", "beta"]:
        p = P[model]
        res[f"{model}_overall"] = sorted([sp(p, qA), sp(p, qB)])
        pc = center_by(p, mats)
        res[f"{model}_centered"] = sorted([sp(pc, cA), sp(pc, cB)])

    # --- branch log loss ---------------------------------------------
    ll = {k: logloss(P[k][:, None], Y).mean(axis=1) for k in P}  # per state
    res["logloss"] = {k: float(v.mean()) for k, v in ll.items()}
    gain_state = ll["direct"] - ll["full"]
    res["logloss_gain_full_vs_direct"] = float(gain_state.mean())
    boot = matrix_boot(lambda i: gain_state[i].mean(), mats, rng)
    res["logloss_gain_lower95"] = float(np.quantile(boot, 0.025))

    # --- q-Brier ------------------------------------------------------
    def qbrier(p, i=None):
        idx = slice(None) if i is None else i
        return 0.5 * (((p[idx] - qA[idx]) ** 2).mean()
                      + ((p[idx] - qB[idx]) ** 2).mean())
    res["qbrier"] = {k: float(qbrier(P[k])) for k in P}
    res["qbrier_gain_full_vs_direct"] = float(
        qbrier(P["direct"]) - qbrier(P["full"]))
    boot = matrix_boot(
        lambda i: qbrier(P["direct"], i) - qbrier(P["full"], i), mats, rng)
    res["qbrier_gain_lower95"] = float(np.quantile(boot, 0.025))

    # --- whole-matrix permutations -----------------------------------
    # Reassign each matrix's block of frozen predictions to a permuted
    # matrix's block of measured q values (matrices with all 5 states).
    counts = {m: np.sum(mats == m) for m in np.unique(mats)}
    full_mats = [m for m, c in counts.items() if c == 5]
    keep = np.isin(mats, full_mats)
    order = np.lexsort((np.array([r["landmark"] for r in rows])[keep],
                        mats[keep]))
    q_blocks = q[keep][order].reshape(len(full_mats), 5)
    perm_p = {}
    for model in ["full", "direct"]:
        p_blocks = P[model][keep][order].reshape(len(full_mats), 5)
        obs = sp(p_blocks.ravel(), q_blocks.ravel())
        exceed = 0
        prng = np.random.default_rng(913)
        for _ in range(N_PERM):
            pi = prng.permutation(len(full_mats))
            if sp(p_blocks[pi].ravel(), q_blocks.ravel()) >= obs:
                exceed += 1
        perm_p[model] = (1 + exceed) / (N_PERM + 1)
    res["permutation_p"] = perm_p
    res["n_full_matrices"] = len(full_mats)

    # --- process probabilities ---------------------------------------
    proc = {}
    allp = {k: np.concatenate([r["proc"][k] for r in rows])
            for k in rows[0]["proc"]}
    brk = allp["break"]
    proc["break"] = float(np.nanmean(brk))
    for k in ["resume2", "episode3", "persist5"]:
        proc[k] = float(np.nanmean(allp[k][brk == 1]))
    proc["old_return"] = float(np.nanmean(allp["old_return"]))
    proc["pos_gain"] = float(np.nanmean(allp["pos_gain"]))
    proc["mean_gain"] = float(np.nanmean(allp["gain"]))
    proc["repeat"] = float(np.nanmean(allp["repeat"]))
    res["process"] = proc

    # per-matrix bootstrap CIs for the bar chart
    per_mat = {}
    for k in ["break", "resume2", "episode3", "old_return",
              "persist5", "pos_gain", "repeat"]:
        vals = []
        for m in np.unique(mats):
            rs = [r for r in rows if r["matrix"] == m]
            v = np.concatenate([r["proc"][k] for r in rs])
            if k in ("resume2", "episode3", "persist5"):
                b = np.concatenate([r["proc"]["break"] for r in rs])
                v = v[b == 1]
            elif k == "pos_gain":
                v = v[~np.isnan(v)]
            vals.append(np.nanmean(v) if len(v) else np.nan)
        per_mat[k] = np.array(vals)
    res["_per_mat_proc"] = per_mat

    res["_arrays"] = {"qA": qA, "qB": qB, "q": q, "mats": mats, "P": P,
                      "cA": cA, "cB": cB}
    return res


def make_figures(results):
    os.makedirs(FIG, exist_ok=True)

    # --- rank transfer bars (mirror of paper Figure 8) ---------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    models = ["direct", "beta", "full"]
    labels = ["history", "beta", "full state"]
    for j, cand in enumerate(CANDIDATES):
        r = results[cand]
        for i, scope in enumerate(["overall", "centered"]):
            ax = axes[i, j]
            vals = [np.mean(r[f"{m}_{scope}"]) for m in models]
            ax.bar(labels, vals, color=BLUE, width=0.62)
            ax.axhline(0, color=INK, lw=0.8)
            scope_name = "overall" if scope == "overall" else "within matrix"
            ax.set_title(f"Candidate {cand} — {scope_name}")
            ax.set_ylabel("Spearman")
            ax.grid(axis="x", visible=False)
    fig.suptitle("Frozen model ranking on untouched matrices (replication)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_rank_transfer.png"))
    plt.close(fig)

    # --- calibration scatter (mirror of paper Figure 9) --------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    for j, cand in enumerate(CANDIDATES):
        a = results[cand]["_arrays"]
        ax = axes[j]
        ax.scatter(a["P"]["full"], a["qB"], s=12, alpha=0.55,
                   color=BLUE, edgecolors="none")
        ax.plot([0, 1], [0, 1], "--", color=INK, lw=0.9)
        ax.set_title(f"Candidate {cand}")
        ax.set_xlabel("Frozen past-observable prediction")
        if j == 0:
            ax.set_ylabel("Independent branch-half probability")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    fig.suptitle("Prospective frozen-coordinate calibration view (replication)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_calibration.png"))
    plt.close(fig)

    # --- process prevalences (mirror of paper Figure 7) --------------
    keys = ["break", "resume2", "episode3", "old_return",
            "persist5", "pos_gain", "repeat"]
    names = ["break", "resume-2", "episode-3", "old-return",
             "persist-5", "positive-gain", "repeat-return"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    xpos, vals, errs, ticks = [], [], [], []
    x = 0
    for cand in CANDIDATES:
        r = results[cand]
        for k, nm in zip(keys, names):
            pm = r["_per_mat_proc"][k]
            pm = pm[~np.isnan(pm)]
            boots = np.array([
                np.mean(np.random.default_rng(7 + i).choice(pm, len(pm)))
                for i in range(400)])
            xpos.append(x)
            vals.append(r["process"][k] if k != "pos_gain"
                        else r["process"]["pos_gain"])
            errs.append([max(vals[-1] - np.quantile(boots, 0.025), 0),
                         max(np.quantile(boots, 0.975) - vals[-1], 0)])
            ticks.append(f"CONF-{cand}\n{nm}")
            x += 1
        x += 1
    ax.bar(xpos, vals, yerr=np.array(errs).T, color=BLUE, width=0.7,
           error_kw={"elinewidth": 0.9, "ecolor": INK})
    ax.set_xticks(xpos)
    ax.set_xticklabels(ticks, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Branch probability")
    ax.grid(axis="x", visible=False)
    ax.set_title("Seven distinct heredity-process probabilities (replication)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_process_prevalence.png"))
    plt.close(fig)

    # --- reliability scatter -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    for j, cand in enumerate(CANDIDATES):
        a = results[cand]["_arrays"]
        ax = axes[j]
        ax.scatter(a["qA"], a["qB"], s=12, alpha=0.55, color=BLUE,
                   edgecolors="none")
        ax.plot([0, 1], [0, 1], "--", color=INK, lw=0.9)
        r = results[cand]
        ax.set_title(f"Candidate {cand} — Spearman "
                     f"{r['reliability']:.3f}")
        ax.set_xlabel("Branch half A (32 futures)")
        if j == 0:
            ax.set_ylabel("Branch half B (32 futures)")
    fig.suptitle("Independent branch-half F12 probability (replication)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_reliability.png"))
    plt.close(fig)


def main():
    with open(os.path.join(OUT, "conf_data.pkl"), "rb") as f:
        data = pickle.load(f)
    results = {}
    for cand in CANDIDATES:
        rows = [r for r in data["table"] if r["candidate"] == cand]
        results[cand] = analyze_candidate(rows)

    make_figures(results)

    clean = {}
    for cand, r in results.items():
        clean[cand] = {k: v for k, v in r.items()
                       if not k.startswith("_")}
    clean["replay"] = data["replay"]
    clean["frozen_models_sha256"] = data["frozen_models_sha256"]
    clean["n_dead_trajectories"] = data["n_dead_trajectories"]
    with open(os.path.join(OUT, "confirmation_metrics.json"), "w") as f:
        json.dump(clean, f, indent=2)

    for cand in CANDIDATES:
        r = results[cand]
        print(f"\n=== Candidate {cand} ===")
        print(f"states: {r['n_states']} | q in (0.1,0.9): "
              f"{r['q_transition_count']}/{r['n_states']}")
        print(f"reliability qA~qB: {r['reliability']:.3f} "
              f"(lower95 {r['reliability_lower95']:.3f}) | centered "
              f"{r['reliability_centered']:.3f} "
              f"(lower95 {r['reliability_centered_lower95']:.3f})")
        for m in ["full", "direct", "beta"]:
            o, c = r[f"{m}_overall"], r[f"{m}_centered"]
            print(f"{m:7s} overall [{o[0]:.3f},{o[1]:.3f}] | "
                  f"centered [{c[0]:.3f},{c[1]:.3f}]")
        print(f"logloss gain full-vs-direct: "
              f"{r['logloss_gain_full_vs_direct']:.4f} "
              f"(lower95 {r['logloss_gain_lower95']:.6f})")
        print(f"q-Brier gain: {r['qbrier_gain_full_vs_direct']:.4f} "
              f"(lower95 {r['qbrier_gain_lower95']:.6f})")
        print(f"permutation p: {r['permutation_p']}")
        print("process:", {k: round(v, 4) for k, v in r["process"].items()})
    print("\nreplay:", data["replay"]["ok"])


if __name__ == "__main__":
    main()
