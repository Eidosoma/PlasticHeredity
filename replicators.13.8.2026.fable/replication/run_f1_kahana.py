"""Phase F1: in-model reproduction of the Kahana-Segev-Lancet
attractor protocol (their MATLAB is unrunnable here — feasibility gate
resolved to protocol-reimplementation mode; their simulator is the
same historical GARD10 source our candidate 02 was validated against).

Config A ("kahana"): candidate-02 kinetics at their configuration —
  splitsize 1.0 (nmax=100, nmin=50), random multinomial mass-50 initial
  compositions (their histc(rand*NG) recipe), 30 generations per run,
  20 initial compositions per matrix, 12 fresh matrices
  (tag f1-kahana-2026-08-13).
Config B ("frozen"): candidates 02 and 03 at the frozen nmax=80, the
  24 steering matrices, same protocol.

Per run, on the within-growth mass grid (step 5):
  - composome similarity (nearest atlas center) and R_Q traces;
  - convergence: fraction of runs reaching sim >= 0.9 by generation 30
    and time-to-first-crossing;
  - within-growth return (Figure 5B analog): mean [sim(pre-fission) -
    sim(post-fission start)] per generation, registered margin +0.05
    with matrix-bootstrap CI excluding 0, with rising R_Q;
  - fission displacement: mean [sim(next post-fission) -
    sim(pre-fission)] (the natural perturbation).

Also builds and persists the composome atlases used by all later
Phase F stages (results_f/atlases.pkl).
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
import cohort
import atlas as AT
import growth_trace as GT
import run_intervention as RI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_f")
FIG = os.path.join(OUT, "figures")
KAHANA_TAG = "f1-kahana-2026-08-13"
STEER_TAG = "steering-2026-08-13"
N_KMAT = 12
N_INITS = 20
N_GENS = 30
N_SMAT = 24
GRID = 5
N_BOOT = 1024

BLUE = "#4878A8"
AMBER = "#A8641E"
GRAY = "#8A939B"
INK = "#33383D"

CONFIGS = [
    ("kahana", "02", 100, KAHANA_TAG, N_KMAT),
    ("frozen02", "02", 80, STEER_TAG, N_SMAT),
    ("frozen03", "03", 80, STEER_TAG, N_SMAT),
]


def atlas_unit(args):
    cfg, cand, nmax, tag, m = args
    ent = cohort.domain_entropy("confirmation", tag) \
        if tag == STEER_TAG else cohort.domain_entropy("dev", tag)
    a = AT.build_atlas(m, cand, ent, nmax=nmax)
    return (cfg, m, a)


def kahana_unit(args):
    cfg, cand, nmax, tag, m, atl = args
    ent = cohort.domain_entropy("confirmation", tag) \
        if tag == STEER_TAG else cohort.domain_entropy("dev", tag)
    beta, _ = cohort.matrix_and_init(ent, m)
    cand_i = cohort.CANDIDATES.index(cand)
    nmin = nmax // 2
    runs = []
    for ic in range(N_INITS):
        rng = cohort._rng(ent, 17, cand_i, m, ic)   # domain 17: F1 runs
        n0 = np.bincount(rng.integers(0, sim.NG, nmin),
                         minlength=sim.NG).astype(np.int64)
        out = GT.traced_run_fissions(n0, beta, cand, N_GENS, rng,
                                     nmax, grid_step=GRID)
        gens = []
        prev_parent = None
        for r in out["recs"]:
            start_m, start_c = r["snaps"][0]
            end_m, end_c = r["snaps"][-1]
            sim_start = AT.nearest_sim(start_c, atl)
            sim_end = AT.nearest_sim(end_c, atl)
            gens.append({
                "sim_start": sim_start, "sim_end": sim_end,
                "rq_start": GT.r_q(start_c, beta),
                "rq_end": GT.r_q(end_c, beta),
                "split_disp": (AT.nearest_sim(r["daughter"], atl)
                               - sim_end),
                "grid": [(mm, AT.nearest_sim(cc, atl),
                          GT.r_q(cc, beta)) for mm, cc in r["snaps"]],
            })
            prev_parent = r["parent"]
        sims_end = [g["sim_end"] for g in gens]
        conv_gen = next((i + 1 for i, s in enumerate(sims_end)
                         if s >= 0.9), None)
        runs.append({"gens": gens, "converged": conv_gen is not None,
                     "conv_gen": conv_gen})
    return {"cfg": cfg, "matrix": m, "runs": runs}


def main():
    os.makedirs(FIG, exist_ok=True)

    # ---- atlases (persisted for all later F stages) ------------------
    t0 = time.time()
    jobs = []
    for cfg, cand, nmax, tag, nm in CONFIGS:
        for m in range(nm):
            jobs.append((cfg, cand, nmax, tag, m))
    with Pool(12) as pool:
        res = pool.map(atlas_unit, jobs)
    atlases = {}
    for cfg, m, a in res:
        atlases[(cfg, m)] = a
    with open(os.path.join(OUT, "atlases.pkl"), "wb") as f:
        pickle.dump(atlases, f, protocol=4)
    ks = {}
    for cfg, cand, nmax, tag, nm in CONFIGS:
        kk = [atlases[(cfg, m)]["k"] for m in range(nm)]
        ks[cfg] = kk
        print(f"atlases {cfg}: k distribution {np.bincount(kk, minlength=7)[1:]}")
    print(f"atlases built in {time.time()-t0:.0f}s")

    # ---- Kahana protocol runs ---------------------------------------
    t0 = time.time()
    jobs = []
    for cfg, cand, nmax, tag, nm in CONFIGS:
        for m in range(nm):
            jobs.append((cfg, cand, nmax, tag, m, atlases[(cfg, m)]))
    with Pool(12) as pool:
        units = pool.map(kahana_unit, jobs)
    print(f"F1 runs in {time.time()-t0:.0f}s")

    results = {"atlas_k": {c: list(map(int, v)) for c, v in ks.items()}}
    for cfg, cand, nmax, tag, nm in CONFIGS:
        cu = [u for u in units if u["cfg"] == cfg]
        mats, wg, rqg, disp, conv, tconv = [], [], [], [], [], []
        for u in cu:
            for run in u["runs"]:
                g5plus = run["gens"][4:]     # post-transient (gen >= 5)
                if not g5plus:
                    continue
                mats.append(u["matrix"])
                wg.append(np.mean([g["sim_end"] - g["sim_start"]
                                   for g in g5plus]))
                rqg.append(np.mean([g["rq_end"] - g["rq_start"]
                                    for g in g5plus]))
                disp.append(np.mean([g["split_disp"] for g in g5plus]))
                conv.append(run["converged"])
                if run["conv_gen"] is not None:
                    tconv.append(run["conv_gen"])
        mats = np.array(mats)
        wg = np.array(wg)
        rng = np.random.default_rng(777)
        ci = RI.boot_lower(wg, mats, rng, n=N_BOOT)
        entry = {
            "n_runs": int(len(wg)),
            "within_growth_gain": {"mean": float(wg.mean()), "ci": ci},
            "rq_gain": float(np.mean(rqg)),
            "fission_displacement": float(np.mean(disp)),
            "converged_frac": float(np.mean(conv)),
            "median_conv_gen": float(np.median(tconv)) if tconv else None,
            "margin_pass": bool(wg.mean() >= 0.05 and ci[0] > 0
                                and np.mean(rqg) > 0),
        }
        results[cfg] = entry
        print(f"\n=== F1 {cfg} ===")
        print(f"within-growth composome gain {wg.mean():+.4f} "
              f"CI [{ci[0]:+.4f},{ci[1]:+.4f}] | R_Q gain "
              f"{np.mean(rqg):+.4f} | fission displacement "
              f"{np.mean(disp):+.4f}")
        print(f"converged (>=0.9 by gen 30): {np.mean(conv):.2f} | "
              f"median conv gen {entry['median_conv_gen']} | "
              f"REGISTERED MARGIN PASS: {entry['margin_pass']}")

    with open(os.path.join(OUT, "f1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figure: phase profiles -------------------------------------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                         "axes.titlecolor": INK})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, (cfg, cand, nmax, tag, nm) in zip(axes, CONFIGS):
        cu = [u for u in units if u["cfg"] == cfg]
        prof = {}
        for u in cu:
            for run in u["runs"]:
                for g in run["gens"][4:]:
                    for mm, s, q in g["grid"]:
                        prof.setdefault(mm, []).append((s, q))
        masses = sorted(prof)
        s_mean = [np.mean([v[0] for v in prof[mm]]) for mm in masses]
        q_mean = [np.mean([v[1] for v in prof[mm]]) for mm in masses]
        frac = [(mm - masses[0]) / (masses[-1] - masses[0])
                for mm in masses]
        ax.plot(frac, s_mean, color=BLUE, lw=1.5,
                label="composome similarity")
        ax2 = ax.twinx()
        ax2.plot(frac, q_mean, color=AMBER, lw=1.2, ls="--",
                 label="R_Q")
        ax.set_title(f"{cfg} (nmax={nmax})")
        ax.set_xlabel("growth phase (mass fraction)")
        if cfg == "kahana":
            ax.set_ylabel("nearest-composome similarity", color=BLUE)
        ax2.set_ylabel("R_Q", color=AMBER)
    fig.suptitle("F1: within-growth composome attraction and "
                 "composition-flux alignment (Kahana Figure 5B analog)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_f1_phase_profiles.png"))
    plt.close(fig)
    print("\nwritten:", os.path.join(OUT, "f1_results.json"))


if __name__ == "__main__":
    main()
