"""Step 3: parameter-regime probe.

Registered perturbed regimes (NG/nmin/nmax/Kf/Kb unchanged):
    (A, sigma) in {(-4, 3), (-4, 5), (-3, 4), (-5, 4)}

Per regime, per candidate: 20 development matrices (train a
regime-matched v2-style coordinate via registry_v2.train_v2) and 20
untouched confirmation matrices with 5 landmarks x 32 branches in
halves of 16. Reports:
  - phenomenology: break, resumption|break, episode3|break, old-return
    prevalence, mean anchor gain (does plastic heredity persist?)
  - split-half q reliability
  - regime-matched v2 vs direct-8 (overall + matrix-centered Spearman)
  - zero-training transfer of the FROZEN main v2 (secondary row)

A probe, not a confirmation: 20+20 matrices per regime.
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

import sim
import features as Ft
import cohort
import registry_v2 as R2
from run_ablation import center_by

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_regimes")
REGIMES = [(-4.0, 3.0), (-4.0, 5.0), (-3.0, 4.0), (-5.0, 4.0)]
N_DEV = 20
N_CONF = 20
N_BRANCHES = 32
HALF = 16
N_WORKERS = 12

# module globals set per regime before each Pool (inherited via fork)
_A = sim.A_MU
_S = sim.SIGMA
_ENT_DEV = 0
_ENT_CONF = 0


def _beta_init(entropy, m):
    beta = sim.make_beta(cohort._rng(entropy, 0, m), a_mu=_A, sigma=_S)
    n0 = sim.make_initial_state(cohort._rng(entropy, 1, m))
    return beta, n0


def regime_dev_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = _beta_init(_ENT_DEV, m)
    rng = cohort._rng(_ENT_DEV, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, cohort.N_FISSIONS, rng)
    nd = traj["n_done"]
    hs, inh, daughters = traj["H"], traj["inherited"], traj["daughters"]
    rows9, rows195, ys = [], [], []
    for g in range(1, nd - cohort.HORIZON + 1):
        state = daughters[g - 1]
        rows9.append(Ft.direct9(g, cohort.N_FISSIONS, hs[:g],
                                int(state.sum())))
        rows195.append(Ft.graph_state_195(state, beta))
        ys.append(float(Ft.joint_break_run3(inh[g:g + cohort.HORIZON])))
    return {"candidate": cand,
            "X9": np.array(rows9).reshape(len(rows9), 9),
            "X195": np.array(rows195).reshape(len(rows195), 195),
            "y": np.array(ys)}


def regime_conf_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = _beta_init(_ENT_CONF, m)
    rng = cohort._rng(_ENT_CONF, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, cohort.N_FISSIONS, rng)
    nd = traj["n_done"]
    hs, daughters = traj["H"], traj["daughters"]
    states = []
    for lm in cohort.LANDMARKS:
        if lm > nd:
            continue
        restored = daughters[lm - 1]
        x9 = Ft.direct9(lm, cohort.N_FISSIONS, hs[:lm], int(restored.sum()))
        x195 = Ft.graph_state_195(restored, beta)
        yb = np.zeros(N_BRANCHES)
        proc = {k: [] for k in cohort.PROC_KEYS}
        for b in range(N_BRANCHES):
            rb = cohort._rng(_ENT_CONF, 3, cand_i, m, lm, b)
            br = sim.run_fissions(restored, beta, cand, cohort.HORIZON, rb)
            inh = br["inherited"]
            yb[b] = float(Ft.joint_break_run3(inh))
            po = Ft.process_outcomes(inh, br["daughters"], restored)
            for k in cohort.PROC_KEYS:
                proc[k].append(po[k])
        states.append({"matrix": m, "landmark": lm, "X9": x9, "X195": x195,
                       "yb": yb, "qA": yb[:HALF].mean(),
                       "qB": yb[HALF:].mean(),
                       "proc": {k: np.array(v, dtype=float)
                                for k, v in proc.items()}})
    return {"candidate": cand, "matrix": m, "states": states,
            "died": traj["died"]}


def main():
    global _A, _S, _ENT_DEV, _ENT_CONF
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        main_v2 = pickle.load(f)

    results = {}
    for A, S in REGIMES:
        key = f"A{A:g}_S{S:g}"
        _A, _S = A, S
        _ENT_DEV = cohort.domain_entropy("dev", f"regime-{key}-2026-08-13")
        _ENT_CONF = cohort.domain_entropy("confirmation",
                                          f"regime-{key}-2026-08-13")
        t0 = time.time()
        jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_DEV)]
        with Pool(N_WORKERS) as pool:
            dev = pool.map(regime_dev_unit, jobs)
        jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_CONF)]
        with Pool(N_WORKERS) as pool:
            conf = pool.map(regime_conf_unit, jobs)
        print(f"regime {key}: simulated in {time.time()-t0:.0f}s")

        results[key] = {}
        for cand in cohort.CANDIDATES:
            du = [u for u in dev if u["candidate"] == cand
                  and len(u["y"]) > 0]
            if not du:
                results[key][cand] = {"error": "no dev data"}
                continue
            X9 = np.vstack([u["X9"] for u in du])
            X195 = np.vstack([u["X195"] for u in du])
            y = np.concatenate([u["y"] for u in du])
            if 0 < y.mean() < 1:
                bundle = R2.train_v2(X9, X195, y)
            else:
                bundle = None

            rows = [s for u in conf if u["candidate"] == cand
                    for s in u["states"]]
            if len(rows) < 20:
                results[key][cand] = {"error": "insufficient conf states"}
                continue
            mats = np.array([r["matrix"] for r in rows])
            qA = np.array([r["qA"] for r in rows])
            qB = np.array([r["qB"] for r in rows])
            cX9 = np.stack([r["X9"] for r in rows])
            cX195 = np.stack([r["X195"] for r in rows])

            def ranks(p):
                pc = center_by(p, mats)
                return {
                    "overall": float(np.mean([
                        spearmanr(p, qA).correlation,
                        spearmanr(p, qB).correlation])),
                    "centered": float(np.mean([
                        spearmanr(pc, center_by(qA, mats)).correlation,
                        spearmanr(pc, center_by(qB, mats)).correlation])),
                }

            entry = {
                "n_states": len(rows),
                "dev_prevalence": float(y.mean()),
                "q_mean": float((qA + qB).mean() / 2),
                "reliability": float(spearmanr(qA, qB).correlation),
            }
            if bundle is not None:
                p = R2.predict_v2(bundle, cX9, cX195)
                entry["matched_v2"] = ranks(p["v2"])
                entry["direct8"] = ranks(p["direct8"])
            pf = R2.predict_v2(main_v2[cand], cX9, cX195)
            entry["frozen_main_v2"] = ranks(pf["v2"])

            allp = {k: np.concatenate([r["proc"][k] for r in rows])
                    for k in cohort.PROC_KEYS}
            brk = allp["break"]
            entry["process"] = {
                "break": float(np.nanmean(brk)),
                "resume2_given_break": float(np.nanmean(
                    allp["resume2"][brk == 1])) if (brk == 1).any() else None,
                "episode3_given_break": float(np.nanmean(
                    allp["episode3"][brk == 1])) if (brk == 1).any() else None,
                "old_return": float(np.nanmean(allp["old_return"])),
                "mean_gain": float(np.nanmean(allp["gain"])),
            }
            results[key][cand] = entry
            mv = entry.get("matched_v2", {}).get("centered")
            d8 = entry.get("direct8", {}).get("centered")
            print(f"  {key} cand {cand}: rel {entry['reliability']:.3f} | "
                  f"q_mean {entry['q_mean']:.3f} | matched-v2 ctr "
                  f"{mv if mv is None else round(mv,3)} vs d8 "
                  f"{d8 if d8 is None else round(d8,3)} | frozen-v2 ctr "
                  f"{entry['frozen_main_v2']['centered']:.3f} | "
                  f"break {entry['process']['break']:.2f}")

    with open(os.path.join(OUT, "regimes_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten:", os.path.join(OUT, "regimes_results.json"))


if __name__ == "__main__":
    main()
