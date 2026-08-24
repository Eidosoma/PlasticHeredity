"""Phase F7: the attractor-aware controller.

Question (registered): was Phase E's evaporation the chemistry's fault,
or the objective's? The original controller minimized break-risk; the
Kahana framing suggests the property to write is alignment with the
endogenous attractor (high R_Q, high composome similarity).

Controllers (per-fission mass-preserving swap on the daughter; lineage
streams domain 16, controller NOT in the key):
  v2_down    : minimize frozen v2 break-risk (the Phase C controller)
  rq_only    : maximize R_Q (composition-flux alignment)
  comp_only  : maximize nearest-atlas-composome similarity
  joint      : maximize J = (1 - risk) + R_Q + atlas_sim (registered
               equal weights)
  random     : uniformly random legal swap (domain 16 sub-stream)
  noop       : none

Design: 24 matrices x 2 candidates x reps {0,1}; 60 steering fissions;
then 60 RELEASE fissions (no edits), tracking similarity to the
controller's own final written composition and to the atlas.

Registered primary comparison: mean anchor similarity at release
fissions {5, 10, 20, 60} and composition-hold fraction (end anchor
H > 0.9) per controller; the attractor-aware claim requires joint or
comp_only to exceed v2_down's anchor similarity at release+10 by
>= 0.10 with matrix-bootstrap CI > 0.

Registered prediction (recorded before running): no controller achieves
durable post-release hold; attractor-aware states may decay marginally
slower initially; "steering wheel, not programmer" is maximally
strengthened if even the joint controller's state evaporates.
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
import atlas as AT
import growth_trace as GT
import run_intervention as RI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_f")
FIG = os.path.join(OUT, "figures")
TAG = "steering-2026-08-13"
N_MAT = 24
REPS = [0, 1]
STEER = 60
RELEASE = 60
CONTROLLERS = ["v2_down", "rq_only", "comp_only", "joint",
               "random", "noop"]
CHECK = [5, 10, 20, 60]
N_BOOT = 1024

BLUE = "#4878A8"
AMBER = "#A8641E"
GRAY = "#8A939B"
GREEN = "#3E7D5B"
PURPLE = "#7C5CA8"
INK = "#33383D"

_ATLASES = None


def _marginal_scores(n, beta, scorer):
    present = np.where(n > 0)[0]
    eye = np.eye(sim.NG, dtype=np.int64)
    adds = np.array([scorer(n + eye[j]) for j in range(sim.NG)])
    rems = np.array([scorer(n - eye[i]) for i in present])
    return present, adds, rems


def objective_swap(controller, n, beta, cand, atl, X9):
    """Choose the (remove, add) swap for the controller's objective."""
    if controller == "v2_down":
        # minimize v2 risk (vector-scored for efficiency)
        present = np.where(n > 0)[0]
        eye = np.eye(sim.NG, dtype=np.int64)
        add_sc = RI.score_states(RI._BUNDLES[cand], X9,
                                 [Ft.graph_state_195(n + eye[j], beta)
                                  for j in range(sim.NG)])
        rem_sc = RI.score_states(RI._BUNDLES[cand], X9,
                                 [Ft.graph_state_195(n - eye[i], beta)
                                  for i in present])
        j = int(np.argmin(add_sc))
        i = int(present[np.argmin(rem_sc)])
    else:
        if controller == "rq_only":
            scorer = lambda m: GT.r_q(m, beta)
        elif controller == "comp_only":
            scorer = lambda m: AT.nearest_sim(m, atl)
        else:  # joint
            eye = np.eye(sim.NG, dtype=np.int64)
            present = np.where(n > 0)[0]
            add_risk = RI.score_states(
                RI._BUNDLES[cand], X9,
                [Ft.graph_state_195(n + eye[j], beta)
                 for j in range(sim.NG)])
            rem_risk = RI.score_states(
                RI._BUNDLES[cand], X9,
                [Ft.graph_state_195(n - eye[i], beta)
                 for i in present])
            add_J = np.array([(1 - add_risk[j]) + GT.r_q(n + eye[j], beta)
                              + AT.nearest_sim(n + eye[j], atl)
                              for j in range(sim.NG)])
            rem_J = np.array([(1 - rem_risk[k])
                              + GT.r_q(n - eye[i], beta)
                              + AT.nearest_sim(n - eye[i], atl)
                              for k, i in enumerate(present)])
            j = int(np.argmax(add_J))
            i = int(present[np.argmax(rem_J)])
            if i == j:
                j = int(np.argsort(add_J)[::-1][1])
            return (i, j)
        present, adds, rems = _marginal_scores(n, beta, scorer)
        j = int(np.argmax(adds))
        i = int(present[np.argmax(rems)])
    if i == j:
        # next-best add
        order = np.argsort(adds)[::-1] if controller != "v2_down" \
            else np.argsort(add_sc)
        j = int(order[1]) if int(order[0]) == i else int(order[0])
    return (i, j)


def f7_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    cfg = "frozen02" if cand == "02" else "frozen03"
    atl = _ATLASES[(cfg, m)]
    out = {}
    for rep in REPS:
        for ci, ctrl in enumerate(CONTROLLERS):
            rng = cohort._rng(RI._ENT, 16, cand_i, m, rep)
            n = n0.copy()
            hs = []
            for f in range(1, STEER + 1):
                step = sim.run_fissions(n, beta, cand, 1, rng)
                if step["n_done"] < 1:
                    break
                hs.append(float(step["H"][0]))
                n = step["final"]
                if f == STEER or ctrl == "noop":
                    continue
                if ctrl == "random":
                    rr = cohort._rng(RI._ENT, 16, cand_i, m, rep, 99, f)
                    present = np.where(n > 0)[0]
                    i = int(present[rr.integers(len(present))])
                    j = int(rr.integers(sim.NG - 1))
                    if j >= i:
                        j += 1
                    swap = (i, j)
                else:
                    X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
                    swap = objective_swap(ctrl, n, beta, cand, atl, X9)
                n = RI.apply_swap(n, swap)
            written = n.copy()
            X9w = Ft.direct9(len(hs), 100, np.array(hs),
                             int(written.sum()))
            props = {
                "risk": float(RI.score_states(
                    RI._BUNDLES[cand], X9w,
                    [Ft.graph_state_195(written, beta)])[0]),
                "rq": GT.r_q(written, beta),
                "atlas_sim": AT.nearest_sim(written, atl),
                "steer_inherit": float(np.mean(np.array(hs) > 0.9)),
            }
            rr2 = cohort._rng(RI._ENT, 16, cand_i, m, rep, 55, ci)
            rel = sim.run_fissions(written, beta, cand, RELEASE, rr2)
            wa = written.astype(float)
            trace = [sim.cosine_h(d.astype(float), wa)
                     for d in rel["daughters"]]
            props["anchor_at"] = {t: float(trace[t - 1])
                                  if len(trace) >= t else np.nan
                                  for t in CHECK}
            props["comp_hold"] = bool(len(trace) >= RELEASE
                                      and trace[-1] > 0.9)
            props["rel_inherit"] = float(np.mean(rel["inherited"]))
            out[(rep, ctrl)] = props
    return {"matrix": m, "candidate": cand, "out": out}


def main():
    global _ATLASES
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)
    with open(os.path.join(OUT, "atlases.pkl"), "rb") as f:
        _ATLASES = pickle.load(f)

    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MAT)]
    with Pool(12) as pool:
        units = pool.map(f7_unit, jobs)
    print(f"F7 campaign in {time.time()-t0:.0f}s")

    results = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        entry = {}
        for ctrl in CONTROLLERS:
            rows = [u["out"][(rep, ctrl)] for u in cu for rep in REPS]
            entry[ctrl] = {
                "risk": float(np.mean([r["risk"] for r in rows])),
                "rq": float(np.mean([r["rq"] for r in rows])),
                "atlas_sim": float(np.mean([r["atlas_sim"]
                                            for r in rows])),
                "steer_inherit": float(np.mean([r["steer_inherit"]
                                                for r in rows])),
                "rel_inherit": float(np.mean([r["rel_inherit"]
                                              for r in rows])),
                "comp_hold": float(np.mean([r["comp_hold"]
                                            for r in rows])),
                "anchor_at": {t: float(np.nanmean(
                    [r["anchor_at"][t] for r in rows])) for t in CHECK},
            }
        # registered comparison at release+10
        mats = np.array([u["matrix"] for u in cu for _ in REPS])
        def a10(ctrl):
            return np.array([u["out"][(rep, ctrl)]["anchor_at"][10]
                             for u in cu for rep in REPS])
        rng = np.random.default_rng(4711)
        best_aware = max(("joint", "comp_only"),
                         key=lambda c: np.nanmean(a10(c)))
        diff = a10(best_aware) - a10("v2_down")
        ok = np.isfinite(diff)
        ci = RI.boot_lower(diff[ok], mats[ok], rng, n=N_BOOT)
        entry["aware_vs_v2down_at10"] = {
            "best_aware": best_aware, "mean": float(np.nanmean(diff)),
            "ci": ci,
            "pass": bool(np.nanmean(diff) >= 0.10 and ci[0] > 0),
        }
        results[cand] = entry

        print(f"\n=== F7 candidate {cand} ===")
        print(f"{'controller':10s} {'risk':>6s} {'R_Q':>6s} "
              f"{'atlSim':>7s} {'stInh':>6s} {'relInh':>7s} "
              f"{'hold':>5s} | anchor@5/10/20/60")
        for ctrl in CONTROLLERS:
            e = entry[ctrl]
            print(f"{ctrl:10s} {e['risk']:6.3f} {e['rq']:6.3f} "
                  f"{e['atlas_sim']:7.3f} {e['steer_inherit']:6.3f} "
                  f"{e['rel_inherit']:7.3f} {e['comp_hold']:5.2f} | "
                  + "/".join(f"{e['anchor_at'][t]:.2f}" for t in CHECK))
        c = entry["aware_vs_v2down_at10"]
        print(f"registered comparison ({c['best_aware']} - v2_down) at "
              f"release+10: {c['mean']:+.3f} CI "
              f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] -> pass: "
              f"{c['pass']}")

    with open(os.path.join(OUT, "f7_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # figure: release decay per controller
    plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                         "axes.titlecolor": INK})
    colors = {"v2_down": AMBER, "rq_only": GREEN, "comp_only": PURPLE,
              "joint": BLUE, "random": GRAY, "noop": INK}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for j, cand in enumerate(cohort.CANDIDATES):
        ax = axes[j]
        for ctrl in CONTROLLERS:
            ys = [results[cand][ctrl]["anchor_at"][t] for t in CHECK]
            ax.plot(CHECK, ys, "o-", color=colors[ctrl], lw=1.3, ms=4,
                    label=ctrl)
        ax.set_xlabel("release fission")
        ax.set_title(f"Candidate {cand}")
        if j == 0:
            ax.set_ylabel("similarity to own written anchor")
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("F7: post-release decay by controller objective")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_f7_release.png"))
    plt.close(fig)
    print("\nwritten:", os.path.join(OUT, "f7_results.json"))


if __name__ == "__main__":
    main()
