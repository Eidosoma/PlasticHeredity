"""Episode-coherence measurement (reviewer #7), prospectively
registered in the plan before any outcome was seen:

  coherence : span similarity H(d_u, d_{u+2}) > 0.9 for the first
              certified 3-run after the first break
  distinct  : coherence AND H(d_u, anchor) < 0.9
  gates     : (a) P(coherent | joint) >= 0.8 in both candidates;
              (b) frozen v2 centered Spearman on the coherent target
                  exceeds direct-8's with 2,048 whole-matrix-bootstrap
                  95% lower bound of the difference > 0;
              (c) split-half reliability of the coherent-target q >= 0.7.

Part 1: trajectory-level descriptive distributions (span, adjacent
daughter similarities, growth-phase drift, anchor similarity) over all
distinct break->episode events in the regenerated v2-conf trajectories.
Part 2: branch-level targets on the regenerated v2-conf campaign, with
an internal consistency assert against the persisted cohort's H64 data.
"""

import json
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim
import cohort
import registry_v2 as R2
from run_ablation import center_by
from run_sensitivity import break_then_run_q

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_coherence")
FIG = os.path.join(OUT, "figures")
V2_TAG = "v2-conf-2026-08-13"
N_MATRICES = 200
N_WORKERS = 12
HALF = 32

BLUE = "#4878A8"
GRAY = "#8A939B"
INK = "#33383D"


def traj_unit(args):
    """Descriptive tier: enumerate distinct break->episode events in a
    regenerated 100-fission trajectory."""
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(cohort.CONF_ENTROPY, m)
    rng = cohort._rng(cohort.CONF_ENTROPY, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, cohort.N_FISSIONS, rng)
    inh, daughters, parents = (traj["inherited"], traj["daughters"],
                               traj["parents"])
    recs = []
    awaiting = False          # a break has occurred; looking for a 3-run
    anchor_idx = None
    run = 0
    for k in range(len(inh)):
        if not inh[k]:
            if not awaiting:
                awaiting = True
                anchor_idx = k - 1        # daughter before the break
            run = 0
            continue
        if not awaiting:
            continue
        run += 1
        if run == 3:
            u = k - 2
            anchor = (n0 if anchor_idx < 0 else daughters[anchor_idx])
            du = daughters[u].astype(float)
            recs.append({
                "span": sim.cosine_h(du, daughters[u + 2].astype(float)),
                "adj1": sim.cosine_h(du, daughters[u + 1].astype(float)),
                "adj2": sim.cosine_h(daughters[u + 1].astype(float),
                                     daughters[u + 2].astype(float)),
                "growth_drift": sim.cosine_h(du,
                                             parents[u + 1].astype(float))
                if u + 1 < len(parents) else np.nan,
                "anchor_sim": sim.cosine_h(du, anchor.astype(float)),
            })
            awaiting = False
            run = 0
    return {"candidate": cand, "episodes": recs}


def q_stats(rows, key):
    qA = np.array([r["y"][key][:HALF].mean() for r in rows])
    qB = np.array([r["y"][key][HALF:].mean() for r in rows])
    return qA, qB


def main():
    os.makedirs(FIG, exist_ok=True)
    cohort.CONF_ENTROPY = cohort.domain_entropy("confirmation", V2_TAG)
    results = {"registered_gates": {
        "a": "P(coherent|joint) >= 0.8", "b": "v2 centered > d8 centered "
        "on coherent target, boot lower > 0", "c": "reliability >= 0.7"}}

    # ---------------- Part 1: descriptive tier ------------------------
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MATRICES)]
    with Pool(N_WORKERS) as pool:
        tunits = pool.map(traj_unit, jobs)
    print(f"trajectories in {time.time()-t0:.0f}s")
    for cand in cohort.CANDIDATES:
        eps = [e for u in tunits if u["candidate"] == cand
               for e in u["episodes"]]
        arr = {k: np.array([e[k] for e in eps]) for k in eps[0]}
        results[f"descriptive_{cand}"] = {
            "n_episodes": len(eps),
            **{k: {"q10": float(np.nanquantile(v, 0.10)),
                   "median": float(np.nanquantile(v, 0.50)),
                   "q90": float(np.nanquantile(v, 0.90)),
                   "frac_gt_0.9": float(np.nanmean(v > 0.9))}
               for k, v in arr.items()},
        }
        d = results[f"descriptive_{cand}"]
        print(f"cand {cand}: {d['n_episodes']} episodes | span "
              f"med {d['span']['median']:.3f} frac>0.9 "
              f"{d['span']['frac_gt_0.9']:.3f} | growth-drift med "
              f"{d['growth_drift']['median']:.3f} | anchor med "
              f"{d['anchor_sim']['median']:.3f}")

    # ---------------- Part 2: branch target tier ----------------------
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        bunits = pool.map(cohort.conf_coherence_unit, jobs)
    print(f"branch campaign in {time.time()-t0:.0f}s")

    with open(os.path.join(HERE, "results_sensitivity",
                           "v2_cohort.pkl"), "rb") as f:
        vc = pickle.load(f)["table"]
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        bundles = pickle.load(f)
    fmap = {(r["candidate"], r["matrix"], r["landmark"]): r for r in vc}

    span_all = {}
    for cand in cohort.CANDIDATES:
        rows = [s for u in bunits if u["candidate"] == cand
                for s in u["states"]]
        # internal consistency: joint q must match the persisted H64 data
        for s in rows[:50]:
            ref = fmap[(cand, s["matrix"], s["landmark"])]
            ev = break_then_run_q(ref["H64"], ref["lens"], 0.90, 3, 12)
            assert np.array_equal(ev.astype(float), s["y"]["joint"]), \
                (cand, s["matrix"], s["landmark"])
        mats = np.array([r["matrix"] for r in rows])
        X9 = np.stack([fmap[(cand, r["matrix"], r["landmark"])]["X9"]
                       for r in rows])
        X195 = np.stack([fmap[(cand, r["matrix"], r["landmark"])]["X195"]
                         for r in rows])
        p = R2.predict_v2(bundles[cand], X9, X195)
        p_v2, p_d8 = p["v2"], p["direct8"]

        joint_flags = np.concatenate([r["y"]["joint"] for r in rows])
        coh_flags = np.concatenate([r["y"]["coherent"] for r in rows])
        dis_flags = np.concatenate([r["y"]["distinct"] for r in rows])
        spans = np.concatenate([r["span"] for r in rows])
        span_all[cand] = spans[joint_flags == 1]
        cres = {"P_coherent_given_joint": float(
                    coh_flags[joint_flags == 1].mean()),
                "P_distinct_given_coherent": float(
                    dis_flags[coh_flags == 1].mean())}

        rngb = np.random.default_rng(20260813)
        u_mats = np.unique(mats)
        idx_map = {m: np.where(mats == m)[0] for m in u_mats}
        for key in ("joint", "coherent", "distinct"):
            qA, qB = q_stats(rows, key)
            q = (qA + qB) / 2
            cA, cB = center_by(qA, mats), center_by(qB, mats)
            pc_v2 = center_by(p_v2, mats)
            pc_d8 = center_by(p_d8, mats)
            cres[key] = {
                "prevalence": float(q.mean()),
                "reliability": float(spearmanr(qA, qB).correlation),
                "v2_overall": float(np.mean([
                    spearmanr(p_v2, qA).correlation,
                    spearmanr(p_v2, qB).correlation])),
                "v2_centered": float(np.mean([
                    spearmanr(pc_v2, cA).correlation,
                    spearmanr(pc_v2, cB).correlation])),
                "d8_overall": float(np.mean([
                    spearmanr(p_d8, qA).correlation,
                    spearmanr(p_d8, qB).correlation])),
                "d8_centered": float(np.mean([
                    spearmanr(pc_d8, cA).correlation,
                    spearmanr(pc_d8, cB).correlation])),
            }
            if key == "coherent":
                # gate (b): bootstrap lower bound of centered diff
                boot = np.empty(2048)
                for i in range(2048):
                    pick = rngb.choice(u_mats, size=len(u_mats),
                                       replace=True)
                    idx = np.concatenate([idx_map[m] for m in pick])
                    mi = mats[idx]
                    dv = np.mean([
                        spearmanr(center_by(p_v2[idx], mi),
                                  center_by(qA[idx], mi)).correlation,
                        spearmanr(center_by(p_v2[idx], mi),
                                  center_by(qB[idx], mi)).correlation])
                    dd = np.mean([
                        spearmanr(center_by(p_d8[idx], mi),
                                  center_by(qA[idx], mi)).correlation,
                        spearmanr(center_by(p_d8[idx], mi),
                                  center_by(qB[idx], mi)).correlation])
                    boot[i] = dv - dd
                cres["gate_b_diff_lower95"] = float(
                    np.quantile(boot, 0.025))

        cres["gates"] = {
            "a_pass": cres["P_coherent_given_joint"] >= 0.8,
            "b_pass": (cres["coherent"]["v2_centered"]
                       > cres["coherent"]["d8_centered"]
                       and cres["gate_b_diff_lower95"] > 0),
            "c_pass": cres["coherent"]["reliability"] >= 0.7,
        }
        results[cand] = cres
        print(f"\n=== Candidate {cand} ===")
        print(f"P(coherent|joint) = {cres['P_coherent_given_joint']:.3f} | "
              f"P(distinct|coherent) = {cres['P_distinct_given_coherent']:.3f}")
        for key in ("joint", "coherent", "distinct"):
            v = cres[key]
            print(f"{key:9s} prev {v['prevalence']:.3f} rel "
                  f"{v['reliability']:.3f} | v2 ov {v['v2_overall']:.3f} "
                  f"ctr {v['v2_centered']:.3f} | d8 ov "
                  f"{v['d8_overall']:.3f} ctr {v['d8_centered']:.3f}")
        print(f"gate(b) diff lower95: {cres['gate_b_diff_lower95']:+.4f} | "
              f"gates: {cres['gates']}")

    with open(os.path.join(OUT, "coherence_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- figures ----------------------------------------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.titlecolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for j, cand in enumerate(cohort.CANDIDATES):
        ax = axes[j]
        ax.hist(span_all[cand], bins=40, range=(0.5, 1.0), color=BLUE)
        ax.axvline(0.9, color="#A8641E", lw=1.2, ls="--")
        ax.set_title(f"Candidate {cand}")
        ax.set_xlabel("Episode span similarity H(d_u, d_{u+2})")
        if j == 0:
            ax.set_ylabel("Certified branch events")
    fig.suptitle("Coherence of renewal episodes (branch events; "
                 "dashed = registered 0.9 criterion)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_span_similarity.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    keys = ["joint", "coherent", "distinct"]
    for j, cand in enumerate(cohort.CANDIDATES):
        ax = axes[j]
        x = np.arange(3)
        v2v = [results[cand][k]["v2_centered"] for k in keys]
        d8v = [results[cand][k]["d8_centered"] for k in keys]
        ax.bar(x - 0.18, v2v, width=0.34, color=BLUE, label="frozen v2")
        ax.bar(x + 0.18, d8v, width=0.34, color=GRAY, label="direct-8")
        ax.set_xticks(x, ["break-and-\nrenewal", "coherent", "distinct"])
        ax.set_title(f"Candidate {cand}")
        if j == 0:
            ax.set_ylabel("Matrix-centered Spearman")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Frozen coordinate vs direct history across the three "
                 "registered targets")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_coherent_targets.png"))
    plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
