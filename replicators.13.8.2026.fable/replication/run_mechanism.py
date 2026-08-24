"""Phase C3: what is the control knob, physically?

Part 1 (exploratory, labeled): correlate every screened swap's
predicted shift (from results_intervention/selections.pkl) with
registered per-type physical quantities:

  in_boost  b_t = (beta n)_t / N   (how strongly the assembly
                                    catalyses type t's joining)
  out_infl  c_t = sum_i x_i beta_{i,t}  (how strongly type t catalyses
                                         the present assembly)
  self_cat  beta_{t,t}
  rem_count n_i (remove side only)

Swap quantities: add_in_boost, add_out_infl, add_self, rem_in_boost,
rem_out_infl, rem_count.

Rule freeze (registered): the per-type quantity with the largest
|pooled Spearman| against predicted shift on the ADD side, oriented by
its sign, becomes THE rule. rule_up = remove the present type that
minimizes the oriented quantity, add the absent-or-any type that
maximizes it (i != j); rule_down mirrors. If rem_count wins, removal
uses count and addition falls back to the strongest add-side quantity.
The frozen rule is recorded in the JSON before any Part 2 branch runs.

Part 2 (confirmatory): on the SAME home cohort states
(intervention-2026-08-13, regenerated from seeds), arms
{rule_up, rule_down, noop} x 48 CRN branches with the Phase A spawn-key
domain (5, cand_i, m, lm, b). Gates (both candidates):
mean (rule_up - rule_down) > 0 with 1,024 matrix-bootstrap lower
bound > 0. Reported: rule/model efficiency ratio vs Phase A.
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
import run_intervention as RI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_mechanism")
FIG = os.path.join(OUT, "figures")
TAG = "intervention-2026-08-13"
N_MATRICES = 40
N_BRANCHES = 48
N_WORKERS = 12
N_BOOT = 1024

BLUE = "#4878A8"
GRAY = "#8A939B"
INK = "#33383D"

QUANTS = ["in_boost", "out_infl", "self_cat"]
_RULE = None      # frozen after Part 1: (quantity, sign)


def type_quantities(n, beta):
    N = max(int(n.sum()), 1)
    x = n / N
    return {
        "in_boost": (beta @ n) / N,
        "out_infl": x @ beta,          # c_t = sum_i x_i beta[i, t]
        "self_cat": np.diag(beta),
    }


def state_features_unit(args):
    """Regenerate one home-cohort unit's landmark states (n, beta)."""
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    rng = cohort._rng(RI._ENT, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, cohort.N_FISSIONS, rng)
    out = []
    for lm in cohort.LANDMARKS:
        if lm > traj["n_done"]:
            continue
        out.append({"matrix": m, "candidate": cand, "landmark": lm,
                    "n": traj["daughters"][lm - 1], "beta_id": m})
    return {"matrix": m, "candidate": cand, "beta": beta, "states": out}


def rule_swaps(n, beta):
    """Frozen physical rule -> (rule_up, rule_down) swaps. No model."""
    quantity, sign = _RULE
    q = type_quantities(n, beta)[quantity] * sign
    present = np.where(n > 0)[0]
    i_up = int(present[np.argmin(q[present])])      # remove worst
    order = np.argsort(q)[::-1]
    j_up = int(order[0]) if order[0] != i_up else int(order[1])
    i_dn = int(present[np.argmax(q[present])])      # remove best
    order = np.argsort(q)
    j_dn = int(order[0]) if order[0] != i_dn else int(order[1])
    return (i_up, j_up), (i_dn, j_dn)


def rule_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    rng = cohort._rng(RI._ENT, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, cohort.N_FISSIONS, rng)
    bids = list(range(N_BRANCHES))
    states = []
    for lm in cohort.LANDMARKS:
        if lm > traj["n_done"]:
            continue
        n = traj["daughters"][lm - 1]
        up, dn = rule_swaps(n, beta)
        res = {"noop": RI.run_arm(n, beta, cand, cand_i, m, lm, bids),
               "rule_up": RI.run_arm(RI.apply_swap(n, up), beta, cand,
                                     cand_i, m, lm, bids),
               "rule_down": RI.run_arm(RI.apply_swap(n, dn), beta, cand,
                                       cand_i, m, lm, bids)}
        states.append({"matrix": m, "landmark": lm, "arms": res})
    return {"matrix": m, "candidate": cand, "states": states}


def main():
    global _RULE
    os.makedirs(FIG, exist_ok=True)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)
    with open(os.path.join(HERE, "results_intervention",
                           "selections.pkl"), "rb") as f:
        sel_map = pickle.load(f)

    # ---------------- Part 1: correlation table -----------------------
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MATRICES)]
    with Pool(N_WORKERS) as pool:
        sunits = pool.map(state_features_unit, jobs)
    print(f"home states regenerated in {time.time()-t0:.0f}s")

    table = {}
    pooled = {f"add_{q}": [] for q in QUANTS}
    pooled.update({f"rem_{q}": [] for q in QUANTS})
    pooled["rem_count"] = []
    dps = []
    for u in sunits:
        beta = u["beta"]
        for s in u["states"]:
            key = (s["candidate"], s["matrix"], s["landmark"])
            sel = sel_map[key]
            tq = type_quantities(s["n"], beta)
            base = sel["base_score"]
            for (i, j), sc in zip(sel["swaps"], sel["swap_scores"]):
                dps.append(sc - base)
                for q in QUANTS:
                    pooled[f"add_{q}"].append(tq[q][j])
                    pooled[f"rem_{q}"].append(tq[q][i])
                pooled["rem_count"].append(s["n"][i])
    dps = np.array(dps)
    corr = {k: float(spearmanr(np.array(v), dps).correlation)
            for k, v in pooled.items()}
    print("pooled Spearman vs predicted shift:",
          {k: round(v, 3) for k, v in corr.items()})

    # rule freeze: strongest ADD-side per-type quantity
    add_corrs = {q: corr[f"add_{q}"] for q in QUANTS}
    best_q = max(add_corrs, key=lambda q: abs(add_corrs[q]))
    _RULE = (best_q, 1.0 if add_corrs[best_q] > 0 else -1.0)
    print(f"FROZEN RULE: quantity={best_q}, "
          f"orientation={'+' if _RULE[1] > 0 else '-'} "
          f"(pooled rho {add_corrs[best_q]:+.3f})")

    results = {"correlations": corr,
               "frozen_rule": {"quantity": best_q,
                               "orientation": _RULE[1],
                               "pooled_rho": add_corrs[best_q]}}
    with open(os.path.join(OUT, "mechanism_results.json"), "w") as f:
        json.dump(results, f, indent=2)      # rule recorded BEFORE Part 2

    # ---------------- Part 2: rule-controller campaign ----------------
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        runits = pool.map(rule_unit, jobs)
    print(f"rule campaign in {time.time()-t0:.0f}s")

    with open(os.path.join(HERE, "results_intervention",
                           "intervention_results.json")) as f:
        phase_a = json.load(f)["phase_a"]

    for cand in cohort.CANDIDATES:
        rows = [s for u in runits if u["candidate"] == cand
                for s in u["states"]]
        mats = np.array([r["matrix"] for r in rows])
        g = lambda arm: np.array([r["arms"][arm]["q"] for r in rows])
        d = g("rule_up") - g("rule_down")
        rng = np.random.default_rng(999)
        ci = RI.boot_lower(d, mats, rng, n=N_BOOT)
        model_effect = phase_a[cand]["paired_up_down"]["mean"]
        entry = {
            "arm_means": {a: float(g(a).mean())
                          for a in ["rule_up", "noop", "rule_down"]},
            "rule_up_down": {"mean": float(d.mean()), "ci": ci},
            "model_up_down_phase_a": model_effect,
            "efficiency_ratio": float(d.mean() / model_effect),
            "gate": {"direction": bool(d.mean() > 0 and ci[0] > 0)},
        }
        results[cand] = entry
        print(f"cand {cand}: rule up-down {d.mean():+.4f} CI "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}] | model {model_effect:+.4f} "
              f"| efficiency {entry['efficiency_ratio']:.2f} | "
              f"gate {entry['gate']}")

    with open(os.path.join(OUT, "mechanism_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # figures
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.titlecolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    names = list(corr.keys())
    vals = [corr[k] for k in names]
    ax.barh(names, vals, color=[BLUE if k == f"add_{best_q}" else GRAY
                                for k in names])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title("Pooled Spearman: physical quantity vs predicted shift")
    ax = axes[1]
    x = np.arange(2)
    for off, cand, col in [(-0.18, "02", BLUE), (0.18, "03", GRAY)]:
        ax.bar(x + off,
               [results[cand]["rule_up_down"]["mean"],
                results[cand]["model_up_down_phase_a"]],
               width=0.34, color=col, label=f"cand {cand}")
    ax.set_xticks(x, ["physical rule", "frozen model (Phase A)"])
    ax.set_ylabel("up−down effect on q")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Rule controller vs model controller")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_mechanism.png"))
    plt.close(fig)
    print("written:", os.path.join(OUT, "mechanism_results.json"))


if __name__ == "__main__":
    main()
