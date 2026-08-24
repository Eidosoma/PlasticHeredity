"""Phase C2: does the control knob transfer across parameter regimes —
and fail where the theory says it must?

Registered: transfer expected at (A,sigma) in {(-4,5), (-3,4), (-5,4)}
(up-down > 0 with 1,024 matrix-bootstrap lower bound > 0; random-noop
CI includes 0); NULL predicted at (-4,3) (up-down CI includes 0). Edit
selection is zero-shot by the FROZEN home-regime v2. 20 fresh matrices
per regime, landmarks {35, 65}, 4 arms x 48 CRN branches.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim
import features as Ft
import cohort
import run_intervention as RI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_knob_transfer")
FIG = os.path.join(OUT, "figures")
REGIMES = [(-4.0, 5.0), (-3.0, 4.0), (-5.0, 4.0), (-4.0, 3.0)]
NULL_REGIME = (-4.0, 3.0)
N_MATRICES = 20
LANDMARKS = [35, 65]
N_BRANCHES = 48
N_WORKERS = 12
N_BOOT = 1024

BLUE = "#4878A8"
AMBER = "#A8641E"
INK = "#33383D"

_A = _S = 0.0


def transfer_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(cohort._rng(RI._ENT, 0, m), a_mu=_A, sigma=_S)
    n0 = sim.make_initial_state(cohort._rng(RI._ENT, 1, m))
    rng = cohort._rng(RI._ENT, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, cohort.N_FISSIONS, rng)
    nd = traj["n_done"]
    hs, daughters = traj["H"], traj["daughters"]
    bundle = RI._BUNDLES[cand]
    bids = list(range(N_BRANCHES))
    states = []
    for lm in LANDMARKS:
        if lm > nd:
            continue
        n = daughters[lm - 1]
        X9 = Ft.direct9(lm, cohort.N_FISSIONS, hs[:lm], int(n.sum()))
        rr = cohort._rng(RI._ENT, 6, cand_i, m, lm)
        sel = RI.screen_swaps(n, beta, X9, bundle, rr)
        arms = {"noop": None, "up": sel["up"], "down": sel["down"],
                "random": sel["random"]}
        res = {name: RI.run_arm(RI.apply_swap(n, swap), beta, cand,
                                cand_i, m, lm, bids)
               for name, swap in arms.items()}
        states.append({"matrix": m, "landmark": lm, "arms": res})
    return {"matrix": m, "candidate": cand, "states": states}


def main():
    global _A, _S
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)

    results = {}
    for A, S in REGIMES:
        key = f"A{A:g}_S{S:g}"
        _A, _S = A, S
        RI._ENT = cohort.domain_entropy("confirmation",
                                        f"knob-{key}-2026-08-13")
        t0 = time.time()
        jobs = [(m, c) for c in cohort.CANDIDATES
                for m in range(N_MATRICES)]
        with Pool(N_WORKERS) as pool:
            units = pool.map(transfer_unit, jobs)
        print(f"regime {key}: campaign in {time.time()-t0:.0f}s")

        results[key] = {"is_null_regime": (A, S) == NULL_REGIME}
        for cand in cohort.CANDIDATES:
            rows = [s for u in units if u["candidate"] == cand
                    for s in u["states"]]
            mats = np.array([r["matrix"] for r in rows])
            g = lambda arm: np.array([r["arms"][arm]["q"] for r in rows])
            d_ud = g("up") - g("down")
            d_rn = g("random") - g("noop")
            rng = np.random.default_rng(31337)
            ci_ud = RI.boot_lower(d_ud, mats, rng, n=N_BOOT)
            ci_rn = RI.boot_lower(d_rn, mats, rng, n=N_BOOT)
            entry = {
                "n_states": len(rows),
                "arm_means": {a: float(g(a).mean())
                              for a in ["up", "noop", "down", "random"]},
                "up_down": {"mean": float(d_ud.mean()), "ci": ci_ud},
                "random_noop": {"mean": float(d_rn.mean()), "ci": ci_rn},
            }
            if (A, S) == NULL_REGIME:
                entry["gate"] = {"null_confirmed":
                                 bool(ci_ud[0] <= 0 <= ci_ud[1])}
            else:
                entry["gate"] = {
                    "transfer": bool(d_ud.mean() > 0 and ci_ud[0] > 0),
                    "specificity": bool(ci_rn[0] <= 0 <= ci_rn[1]),
                }
            results[key][cand] = entry
            print(f"  {key} cand {cand}: up-down "
                  f"{entry['up_down']['mean']:+.4f} CI "
                  f"[{ci_ud[0]:+.4f},{ci_ud[1]:+.4f}] | random-noop "
                  f"{entry['random_noop']['mean']:+.4f} | "
                  f"gate {entry['gate']}")

    with open(os.path.join(OUT, "knob_transfer_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # figure: up-down effect per regime, null regime highlighted
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.titlecolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    keys = [f"A{A:g}_S{S:g}" for A, S in REGIMES]
    labels = [f"({A:g},{S:g})" + ("\nNULL pred." if (A, S) == NULL_REGIME
                                  else "") for A, S in REGIMES]
    for j, cand in enumerate(cohort.CANDIDATES):
        ax = axes[j]
        means = [results[k][cand]["up_down"]["mean"] for k in keys]
        los = [results[k][cand]["up_down"]["ci"][0] for k in keys]
        his = [results[k][cand]["up_down"]["ci"][1] for k in keys]
        colors = [AMBER if results[k]["is_null_regime"] else BLUE
                  for k in keys]
        x = np.arange(len(keys))
        ax.bar(x, means, color=colors, width=0.6,
               yerr=[np.array(means) - np.array(los),
                     np.array(his) - np.array(means)],
               error_kw={"elinewidth": 1.0, "ecolor": INK})
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(f"Candidate {cand}")
        if j == 0:
            ax.set_ylabel("Paired up−down effect on q")
    fig.suptitle("Knob transfer across regimes (frozen home-regime "
                 "scorer, zero-shot)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_knob_transfer.png"))
    plt.close(fig)
    print("written:", os.path.join(OUT, "knob_transfer_results.json"))


if __name__ == "__main__":
    main()
