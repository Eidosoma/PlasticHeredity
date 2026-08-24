"""From prediction to control: preregistered intervention experiment.

Phase A (paired-extremes gate): per restored state, mass-preserving
single-molecule swaps selected by the FROZEN v2 coordinate — up (max
predicted p̂), down (min), noop, random — each given 64 matched
stochastic futures (common random numbers: the arm is NOT in the branch
spawn key). Gates G1-G3 preregistered in the plan; Phase B
(dose-response over 6 swaps spanning the screened predicted-shift
range) runs only if Phase A passes in both candidates.

Registered spawn key for intervention branches: (5, cand_i, m, lm, b).
Fresh cohort tag: intervention-2026-08-13.
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
import features as Ft
import cohort
import registry_v2 as R2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_intervention")
FIG = os.path.join(OUT, "figures")
TAG = "intervention-2026-08-13"
N_MATRICES = 40
N_BRANCHES = 64
N_WORKERS = 12
TOPK = 10
N_BOOT = 2048

BLUE = "#4878A8"
GRAY = "#8A939B"
AMBER = "#A8641E"
INK = "#33383D"

_BUNDLES = None      # loaded in parent before Pool; inherited via fork
_ENT = None


def score_states(bundle, X9_row, X195_rows):
    X9 = np.tile(X9_row, (len(X195_rows), 1))
    return R2.predict_v2(bundle, X9, np.asarray(X195_rows))["v2"]


def screen_swaps(n, beta, X9_row, bundle, rng_rand):
    """Registered screening: marginal adds/removes -> top-10 each
    direction -> exact scoring of 100 up-up and 100 down-down swaps.
    Returns dict with selected swaps and the screened swap table."""
    base_score = float(score_states(bundle, X9_row,
                                    [Ft.graph_state_195(n, beta)])[0])
    present = np.where(n > 0)[0]

    # marginal single edits (screening only; mass changes by 1 here)
    add_feats = [Ft.graph_state_195(n + np.eye(sim.NG, dtype=np.int64)[j],
                                    beta) for j in range(sim.NG)]
    add_sc = score_states(bundle, X9_row, add_feats)
    rem_feats = [Ft.graph_state_195(n - np.eye(sim.NG, dtype=np.int64)[i],
                                    beta) for i in present]
    rem_sc = score_states(bundle, X9_row, rem_feats)

    up_adds = np.argsort(add_sc)[::-1][:TOPK]
    dn_adds = np.argsort(add_sc)[:TOPK]
    up_rems = present[np.argsort(rem_sc)[::-1][:TOPK]]
    dn_rems = present[np.argsort(rem_sc)[:TOPK]]

    def exact(rems, adds):
        combos, feats = [], []
        for i in rems:
            for j in adds:
                if i == j:
                    continue
                ne = n.copy()
                ne[i] -= 1
                ne[j] += 1
                combos.append((int(i), int(j)))
                feats.append(Ft.graph_state_195(ne, beta))
        return combos, score_states(bundle, X9_row, feats)

    up_c, up_s = exact(up_rems, up_adds)
    dn_c, dn_s = exact(dn_rems, dn_adds)
    swaps = up_c + dn_c
    scores = np.concatenate([up_s, dn_s])

    k_up = int(np.argmax(scores))
    k_dn = int(np.argmin(scores))
    ri = int(present[rng_rand.integers(len(present))])
    rj = int(rng_rand.integers(sim.NG - 1))
    if rj >= ri:
        rj += 1
    return {
        "base_score": base_score,
        "up": swaps[k_up], "up_score": float(scores[k_up]),
        "down": swaps[k_dn], "down_score": float(scores[k_dn]),
        "random": (ri, rj),
        "swaps": swaps, "swap_scores": scores.tolist(),
    }


def apply_swap(n, swap):
    ne = n.copy()
    if swap is not None:
        i, j = swap
        ne[i] -= 1
        ne[j] += 1
        assert ne[i] >= 0
    return ne


def run_arm(state, beta, cand, cand_i, m, lm, branch_ids):
    """64 matched futures for one edited state; CRN: arm not in key."""
    y = np.zeros(len(branch_ids))
    brk = np.zeros(len(branch_ids))
    ep3 = np.full(len(branch_ids), np.nan)
    for ix, b in enumerate(branch_ids):
        rb = cohort._rng(_ENT, 5, cand_i, m, lm, b)
        br = sim.run_fissions(state, beta, cand, cohort.HORIZON, rb)
        inh = br["inherited"]
        y[ix] = float(Ft.joint_break_run3(inh))
        anybrk = bool((~inh).any()) if len(inh) else False
        brk[ix] = float(anybrk)
        if anybrk:
            t = int(np.argmin(inh))
            run = best = 0
            for v in inh[t + 1:]:
                run = run + 1 if v else 0
                best = max(best, run)
            ep3[ix] = float(best >= 3)
    return {"q": y.mean(), "break": brk.mean(),
            "ep3_given_break": float(np.nanmean(ep3))
            if np.isfinite(ep3).any() else np.nan}


def intervention_unit(args):
    m, cand, phase, sel_map = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(_ENT, m)
    rng = cohort._rng(_ENT, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, cohort.N_FISSIONS, rng)
    nd = traj["n_done"]
    hs, daughters = traj["H"], traj["daughters"]
    bundle = _BUNDLES[cand]
    bids = list(range(N_BRANCHES))
    states = []
    for lm in cohort.LANDMARKS:
        if lm > nd:
            continue
        n = daughters[lm - 1]
        X9 = Ft.direct9(lm, cohort.N_FISSIONS, hs[:lm], int(n.sum()))
        if phase == "A":
            rr = cohort._rng(_ENT, 6, cand_i, m, lm)   # random-arm stream
            sel = screen_swaps(n, beta, X9, bundle, rr)
            arms = {"noop": None, "up": sel["up"], "down": sel["down"],
                    "random": sel["random"]}
        else:
            sel = sel_map[(cand, m, lm)]
            scores = np.array(sel["swap_scores"])
            qs = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])
            arms, preds = {}, {}
            for qi, qv in enumerate(qs):
                k = int(np.argmin(np.abs(scores - qv)))
                arms[f"dose{qi}"] = tuple(sel["swaps"][k])
                preds[f"dose{qi}"] = float(scores[k])
        res = {}
        for name, swap in arms.items():
            res[name] = run_arm(apply_swap(n, swap), beta, cand,
                                cand_i, m, lm, bids)
            if phase == "B":
                res[name]["pred"] = preds[name]
        states.append({"matrix": m, "candidate": cand, "landmark": lm,
                       "arms": res, "sel": sel if phase == "A" else None})
    return {"matrix": m, "candidate": cand, "states": states}


def boot_lower(vals, mats, rng, n=N_BOOT, upper=False):
    u = np.unique(mats)
    idx_map = {mm: np.where(mats == mm)[0] for mm in u}
    bs = np.empty(n)
    for i in range(n):
        pick = rng.choice(u, size=len(u), replace=True)
        bs[i] = vals[np.concatenate([idx_map[mm] for mm in pick])].mean()
    return (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))


def analyze_phase_a(units):
    results = {}
    for cand in cohort.CANDIDATES:
        rows = [s for u in units if u["candidate"] == cand
                for s in u["states"]]
        mats = np.array([r["matrix"] for r in rows])
        g = lambda arm, key="q": np.array([r["arms"][arm][key]
                                           for r in rows])
        q_up, q_dn = g("up"), g("down")
        q_no, q_rd = g("noop"), g("random")
        pred_up = np.array([r["sel"]["up_score"] for r in rows])
        pred_dn = np.array([r["sel"]["down_score"] for r in rows])
        pred_base = np.array([r["sel"]["base_score"] for r in rows])
        rng = np.random.default_rng(20260813)

        d_ud = q_up - q_dn
        d_un = q_up - q_no
        d_nd = q_no - q_dn
        d_rn = q_rd - q_no
        ci_ud = boot_lower(d_ud, mats, rng)
        ci_un = boot_lower(d_un, mats, rng)
        ci_nd = boot_lower(d_nd, mats, rng)
        ci_rn = boot_lower(d_rn, mats, rng)

        trans = (q_no > 0.1) & (q_no < 0.9)
        res = {
            "n_states": len(rows),
            "arm_means": {"up": float(q_up.mean()),
                          "noop": float(q_no.mean()),
                          "down": float(q_dn.mean()),
                          "random": float(q_rd.mean())},
            "paired_up_down": {"mean": float(d_ud.mean()), "ci": ci_ud},
            "paired_up_noop": {"mean": float(d_un.mean()), "ci": ci_un},
            "paired_noop_down": {"mean": float(d_nd.mean()), "ci": ci_nd},
            "paired_random_noop": {"mean": float(d_rn.mean()), "ci": ci_rn},
            "transition_subset_up_down": float(d_ud[trans].mean()),
            "predicted_shift_up_down": float(
                (pred_up - pred_dn).mean()),
            "noop_rank_check": float(np.mean([
                spearmanr(pred_base, q_no).correlation])),
            "components": {
                "break_up_down": float((g("up", "break")
                                        - g("down", "break")).mean()),
                "ep3gb_up_down": float(np.nanmean(
                    g("up", "ep3_given_break")
                    - g("down", "ep3_given_break"))),
            },
        }
        res["gates"] = {
            "G1": bool(d_ud.mean() > 0 and ci_ud[0] > 0),
            "G2": bool(q_up.mean() > q_no.mean() > q_dn.mean()),
            "G3": bool(abs(d_rn.mean()) < 0.25 * d_ud.mean()),
        }
        res["pass"] = all(res["gates"].values())
        results[cand] = res
    results["phase_a_pass"] = all(results[c]["pass"]
                                  for c in cohort.CANDIDATES)
    return results


def analyze_phase_b(units_b, sel_map, phase_a_units):
    results = {}
    for cand in cohort.CANDIDATES:
        rows_a = {(s["matrix"], s["landmark"]): s
                  for u in phase_a_units if u["candidate"] == cand
                  for s in u["states"]}
        per_state_rho, pooled_x, pooled_y, mats_list = [], [], [], []
        for u in units_b:
            if u["candidate"] != cand:
                continue
            for s in u["states"]:
                key = (s["matrix"], s["landmark"])
                sa = rows_a[key]
                sel = sel_map[(cand, s["matrix"], s["landmark"])]
                base = sel["base_score"]
                xs = [sel["down_score"]] + \
                     [s["arms"][f"dose{i}"]["pred"] for i in range(4)] + \
                     [sel["up_score"]]
                ys = [sa["arms"]["down"]["q"]] + \
                     [s["arms"][f"dose{i}"]["q"] for i in range(4)] + \
                     [sa["arms"]["up"]["q"]]
                dx = np.array(xs) - base
                dy = np.array(ys) - sa["arms"]["noop"]["q"]
                if np.std(dx) > 0:
                    r = spearmanr(dx, dy).correlation
                    if np.isfinite(r):
                        per_state_rho.append(r)
                        pooled_x.extend(dx)
                        pooled_y.extend(dy)
                        mats_list.append(s["matrix"])
        rho = np.array(per_state_rho)
        mats = np.array(mats_list)
        rng = np.random.default_rng(424242)
        ci_rho = boot_lower(rho, mats, rng)
        px, py = np.array(pooled_x), np.array(pooled_y)
        slope = float(np.polyfit(px, py, 1)[0])
        # matrix bootstrap of the pooled slope
        u_m = np.unique(mats)
        state_of = np.repeat(np.arange(len(rho)), 6)
        bs = np.empty(N_BOOT)
        idx_map = {mm: np.where(mats == mm)[0] for mm in u_m}
        for i in range(N_BOOT):
            pick = rng.choice(u_m, size=len(u_m), replace=True)
            sidx = np.concatenate([idx_map[mm] for mm in pick])
            mask = np.isin(state_of, sidx)
            bs[i] = np.polyfit(px[mask], py[mask], 1)[0]
        results[cand] = {
            "n_states": int(len(rho)),
            "mean_within_state_spearman": float(rho.mean()),
            "spearman_ci": ci_rho,
            "pooled_slope": slope,
            "slope_ci": [float(np.quantile(bs, 0.025)),
                         float(np.quantile(bs, 0.975))],
            "gates": {"B1": bool(rho.mean() > 0 and ci_rho[0] > 0),
                      "B2": bool(slope > 0
                                 and float(np.quantile(bs, 0.025)) > 0)},
        }
        results[cand]["pass"] = all(results[cand]["gates"].values())
    results["phase_b_pass"] = all(results[c]["pass"]
                                  for c in cohort.CANDIDATES)
    return results


def main():
    global _BUNDLES, _ENT
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        _BUNDLES = pickle.load(f)
    _ENT = cohort.domain_entropy("confirmation", TAG)

    t0 = time.time()
    jobs = [(m, c, "A", None) for c in cohort.CANDIDATES
            for m in range(N_MATRICES)]
    with Pool(N_WORKERS) as pool:
        units_a = pool.map(intervention_unit, jobs)
    print(f"Phase A campaign in {time.time()-t0:.0f}s")

    sel_map = {}
    for u in units_a:
        for s in u["states"]:
            sel_map[(s["candidate"], s["matrix"], s["landmark"])] = s["sel"]
    with open(os.path.join(OUT, "selections.pkl"), "wb") as f:
        pickle.dump(sel_map, f, protocol=4)

    res_a = analyze_phase_a(units_a)
    for cand in cohort.CANDIDATES:
        r = res_a[cand]
        print(f"\n=== Phase A candidate {cand} ({r['n_states']} states) ===")
        print("arm means:", {k: round(v, 4)
                             for k, v in r["arm_means"].items()})
        print(f"up-down: {r['paired_up_down']['mean']:+.4f} "
              f"CI {r['paired_up_down']['ci']} | predicted "
              f"{r['predicted_shift_up_down']:+.4f}")
        print(f"up-noop: {r['paired_up_noop']['mean']:+.4f} "
              f"{r['paired_up_noop']['ci']} | noop-down: "
              f"{r['paired_noop_down']['mean']:+.4f} "
              f"{r['paired_noop_down']['ci']}")
        print(f"random-noop: {r['paired_random_noop']['mean']:+.4f} "
              f"{r['paired_random_noop']['ci']}")
        print(f"components: {r['components']} | noop rank check "
              f"{r['noop_rank_check']:.3f}")
        print(f"gates: {r['gates']} -> pass={r['pass']}")
    print(f"\nPHASE A: {'PASS' if res_a['phase_a_pass'] else 'FAIL'}")

    out = {"phase_a": res_a, "tag": TAG}

    if res_a["phase_a_pass"]:
        t0 = time.time()
        jobs = [(m, c, "B", sel_map) for c in cohort.CANDIDATES
                for m in range(N_MATRICES)]
        with Pool(N_WORKERS) as pool:
            units_b = pool.map(intervention_unit, jobs)
        print(f"Phase B campaign in {time.time()-t0:.0f}s")
        res_b = analyze_phase_b(units_b, sel_map, units_a)
        out["phase_b"] = res_b
        for cand in cohort.CANDIDATES:
            r = res_b[cand]
            print(f"\n=== Phase B candidate {cand} ===")
            print(f"within-state Spearman {r['mean_within_state_spearman']:.3f} "
                  f"CI {r['spearman_ci']} | slope {r['pooled_slope']:.3f} "
                  f"CI {r['slope_ci']} | gates {r['gates']}")
        print(f"\nPHASE B: {'PASS' if res_b['phase_b_pass'] else 'FAIL'}")

        # dose-response figure
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), sharey=True)
        for j, cand in enumerate(cohort.CANDIDATES):
            ax = axes[j]
            for u in units_b:
                if u["candidate"] != cand:
                    continue
                for s in u["states"]:
                    key = (cand, s["matrix"], s["landmark"])
                    sel = sel_map[key]
                    sa = [x for uu in units_a if uu["candidate"] == cand
                          for x in uu["states"]
                          if (x["matrix"], x["landmark"]) == key[1:]][0]
                    xs = np.array([sel["down_score"]]
                                  + [s["arms"][f"dose{i}"]["pred"]
                                     for i in range(4)]
                                  + [sel["up_score"]]) - sel["base_score"]
                    ys = np.array([sa["arms"]["down"]["q"]]
                                  + [s["arms"][f"dose{i}"]["q"]
                                     for i in range(4)]
                                  + [sa["arms"]["up"]["q"]]) \
                        - sa["arms"]["noop"]["q"]
                    ax.plot(xs, ys, "-", color=BLUE, alpha=0.08, lw=0.7)
            ax.axhline(0, color=INK, lw=0.6)
            ax.axvline(0, color=INK, lw=0.6)
            ax.set_title(f"Candidate {cand} — slope "
                         f"{res_b[cand]['pooled_slope']:.2f}")
            ax.set_xlabel("Predicted shift in p̂ (frozen v2)")
            if j == 0:
                ax.set_ylabel("Realized shift in branch q")
        fig.suptitle("Dose-response: predicted vs realized probability "
                     "shifts under matched futures")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_dose_response.png"), dpi=150)
        plt.close(fig)

    # Phase A figure: arm means with bootstrap CIs
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), sharey=True)
    for j, cand in enumerate(cohort.CANDIDATES):
        r = res_a[cand]
        ax = axes[j]
        names = ["down", "random", "noop", "up"]
        vals = [r["arm_means"][k] for k in names]
        ax.bar(names, vals, color=[AMBER, GRAY, GRAY, BLUE], width=0.6)
        ax.set_title(f"Candidate {cand} — up−down "
                     f"{r['paired_up_down']['mean']:+.3f}")
        if j == 0:
            ax.set_ylabel("Mean branch q (JOINT_BREAK_RUN3)")
    fig.suptitle("Intervention arms under matched stochastic futures")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_intervention_arms.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "intervention_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwritten:", os.path.join(OUT, "intervention_results.json"))


if __name__ == "__main__":
    main()
