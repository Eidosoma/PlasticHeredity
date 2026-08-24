"""Phase C1: closed-loop steering — does one-shot control compound?

Controllers {model_up, model_down, noop, random} steer lineages for 60
fissions: after every fission, the model controllers score the ~140
marginal single edits with the FROZEN v2, form the swap (best
remove-direction, best add-direction) for their sign, and apply it to
the continuing daughter (no edit on the initial state). `random`
applies a uniformly random legal swap each fission; `noop` none.

CRN: lineage stream spawn key (7, cand_i, m, rep) — controller NOT in
the key (matched initial streams; divergence thereafter is inherent).
Random-controller edits use their own stream (8, cand_i, m, rep, f).

Registered outcomes per lineage: certified break->3-run episode count
over 60 fissions (primary; the run_coherence enumeration), breaks,
inheritance fraction, longest inherited run. Gates (both candidates):
paired (model_up - model_down) episode count > 0 with 1,024
matrix-bootstrap lower bound > 0; ordering model_up > noop >
model_down in point estimates; |random - noop| CI includes 0.

This is the properly-controlled analog of the original paper's
Figure 6 / Table 1 protocol (repeated per-fission edits), which did not
reproduce under the Phi scorer.
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
OUT = os.path.join(HERE, "results_steering")
FIG = os.path.join(OUT, "figures")
TAG = "steering-2026-08-13"
N_MATRICES = 24
N_REPS = 6
HORIZON = 60
N_WORKERS = 12
N_BOOT = 1024
CONTROLLERS = ["model_up", "model_down", "noop", "random"]

BLUE = "#4878A8"
AMBER = "#A8641E"
GRAY = "#8A939B"
GREEN = "#3E7D5B"
INK = "#33383D"


def marginal_swap(n, beta, X9, bundle, sign):
    """Score all single adds and removes; return the (remove, add) swap
    for the controller's sign (+1 = raise score, -1 = lower)."""
    present = np.where(n > 0)[0]
    eye = np.eye(sim.NG, dtype=np.int64)
    add_sc = RI.score_states(bundle, X9,
                             [Ft.graph_state_195(n + eye[j], beta)
                              for j in range(sim.NG)])
    rem_sc = RI.score_states(bundle, X9,
                             [Ft.graph_state_195(n - eye[i], beta)
                              for i in present])
    if sign > 0:
        j = int(np.argmax(add_sc))
        i = int(present[np.argmax(rem_sc)])
    else:
        j = int(np.argmin(add_sc))
        i = int(present[np.argmin(rem_sc)])
    if i == j:
        order = np.argsort(add_sc)[::-1] if sign > 0 else np.argsort(add_sc)
        j = int(order[1]) if int(order[0]) == i else int(order[0])
    return (i, j)


def lineage_stats(inh):
    """Registered outcomes from a lineage's inheritance flags."""
    episodes = 0
    awaiting = False
    run = longest = cur = 0
    breaks = 0
    for v in inh:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
        if not v:
            breaks += 1
            awaiting = True
            run = 0
            continue
        if awaiting:
            run += 1
            if run == 3:
                episodes += 1
                awaiting = False
                run = 0
    return {"episodes": episodes, "breaks": breaks,
            "inherit_frac": float(np.mean(inh)) if len(inh) else 0.0,
            "longest_run": longest}


def steer_lineage(n0, beta, cand, cand_i, m, rep, controller, log=None,
                  horizon=None):
    """Steer one lineage. `log(f, daughter_pre_edit, swap, H, updates)`
    is an optional observer called once per fission BEFORE the edit is
    applied; passing it must not change behavior (no RNG use)."""
    horizon = HORIZON if horizon is None else horizon
    rng = cohort._rng(RI._ENT, 7, cand_i, m, rep)
    bundle = RI._BUNDLES[cand]
    n = n0.copy()
    hs = []
    inh = []
    for f in range(1, horizon + 1):
        step = sim.run_fissions(n, beta, cand, 1, rng)
        if step["n_done"] < 1:
            break
        hs.append(float(step["H"][0]))
        inh.append(bool(step["inherited"][0]))
        n = step["final"]
        swap = None
        if f < horizon and controller != "noop":
            if controller == "random":
                rr = cohort._rng(RI._ENT, 8, cand_i, m, rep, f)
                present = np.where(n > 0)[0]
                i = int(present[rr.integers(len(present))])
                j = int(rr.integers(sim.NG - 1))
                if j >= i:
                    j += 1
                swap = (i, j)
            else:
                X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
                sign = +1 if controller == "model_up" else -1
                swap = marginal_swap(n, beta, X9, bundle, sign)
        if log is not None:
            log(f, n, swap, hs[-1], int(step["updates"][0]))
        if swap is not None:
            n = RI.apply_swap(n, swap)
    return lineage_stats(np.array(inh, dtype=bool))


def steering_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    out = {c: [] for c in CONTROLLERS}
    for rep in range(N_REPS):
        for c in CONTROLLERS:
            out[c].append(steer_lineage(n0, beta, cand, cand_i,
                                        m, rep, c))
    # wired bitwise assert: the noop lineage equals a plain trajectory
    rng = cohort._rng(RI._ENT, 7, cand_i, m, 0)
    plain = sim.run_fissions(n0, beta, cand, HORIZON, rng)
    ref = lineage_stats(plain["inherited"])
    assert ref == out["noop"][0], ("noop-bitwise mismatch", m, cand)
    return {"matrix": m, "candidate": cand, "stats": out}


def main():
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)

    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MATRICES)]
    with Pool(N_WORKERS) as pool:
        units = pool.map(steering_unit, jobs)
    print(f"steering campaign in {time.time()-t0:.0f}s")

    results = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        mats = np.array([u["matrix"] for u in cu])

        def per_matrix(ctrl, key):
            return np.array([np.mean([r[key] for r in u["stats"][ctrl]])
                             for u in cu])

        entry = {"outcome_means": {}}
        for key in ["episodes", "breaks", "inherit_frac", "longest_run"]:
            entry["outcome_means"][key] = {
                c: float(per_matrix(c, key).mean()) for c in CONTROLLERS}

        d_ud = per_matrix("model_up", "episodes") \
            - per_matrix("model_down", "episodes")
        d_rn = per_matrix("random", "episodes") \
            - per_matrix("noop", "episodes")
        rng = np.random.default_rng(4242)
        ci_ud = RI.boot_lower(d_ud, mats, rng, n=N_BOOT)
        ci_rn = RI.boot_lower(d_rn, mats, rng, n=N_BOOT)
        em = entry["outcome_means"]["episodes"]
        entry["episodes_up_down"] = {"mean": float(d_ud.mean()),
                                     "ci": ci_ud}
        entry["episodes_random_noop"] = {"mean": float(d_rn.mean()),
                                         "ci": ci_rn}
        entry["gates"] = {
            "G1_up_down": bool(d_ud.mean() > 0 and ci_ud[0] > 0),
            "G2_ordering": bool(em["model_up"] > em["noop"]
                                > em["model_down"]),
            "G3_random_null": bool(ci_rn[0] <= 0 <= ci_rn[1]),
        }
        entry["pass"] = all(entry["gates"].values())
        results[cand] = entry
        print(f"\n=== Steering candidate {cand} ===")
        for key in ["episodes", "breaks", "inherit_frac", "longest_run"]:
            print(f"{key:13s}: " + " | ".join(
                f"{c} {entry['outcome_means'][key][c]:.3f}"
                for c in CONTROLLERS))
        print(f"episodes up-down {d_ud.mean():+.3f} CI "
              f"[{ci_ud[0]:+.3f},{ci_ud[1]:+.3f}] | random-noop "
              f"{d_rn.mean():+.3f} CI [{ci_rn[0]:+.3f},{ci_rn[1]:+.3f}]")
        print(f"gates {entry['gates']} -> pass={entry['pass']}")
    results["steering_pass"] = all(results[c]["pass"]
                                   for c in cohort.CANDIDATES)
    print(f"\nSTEERING: {'PASS' if results['steering_pass'] else 'FAIL'}")

    with open(os.path.join(OUT, "steering_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # figure: episode counts per controller
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.titlecolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    order = ["model_down", "random", "noop", "model_up"]
    colors = [AMBER, GRAY, GRAY, BLUE]
    for j, cand in enumerate(cohort.CANDIDATES):
        ax = axes[j]
        em = results[cand]["outcome_means"]["episodes"]
        ax.bar(order, [em[c] for c in order], color=colors, width=0.6)
        ax.set_title(f"Candidate {cand} — up−down "
                     f"{results[cand]['episodes_up_down']['mean']:+.2f}")
        if j == 0:
            ax.set_ylabel("Break-and-renewal episodes per 60 fissions")
    fig.suptitle("Closed-loop steering: per-fission single swaps for "
                 "60 generations")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_steering.png"))
    plt.close(fig)
    print("written:", os.path.join(OUT, "steering_results.json"))


if __name__ == "__main__":
    main()
