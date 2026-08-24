"""Phase F2: are controller-written states Kahana-style composomes?

Three prospectively defined state classes per (matrix, candidate), all
at the frozen nmax=80 configuration on the 24 steering matrices:

  natural_composome : post-fission daughters from a fresh probe lineage
                      (domain 18) with atlas similarity >= 0.9 AND
                      R_Q above the probe lineage's median (up to 2
                      per matrix)
  written           : the model_down controller state after the frozen
                      60-fission steering period (Phase E write phase,
                      regenerated; rep 0)
  matched_natural   : the noop lineage's daughter at fission 60

Per state: frozen v2 risk, R_Q, distance to nearest atlas composome,
entropy, occupied types, composition-weighted outgoing catalytic
influence, generating-fission parent-daughter H, and empirical one-step
restoring drift (mean over 16 one-fission branches of
[d_atlas(next daughter) - d_atlas(state)]; domain 19).

Registered key diagnostic: written-state R_Q versus natural-composome
R_Q — a low written R_Q means the controller wrote a low-break-risk
state that is MISALIGNED with the endogenous flux, explaining Phase E's
evaporation without contradicting Kahana.
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

import sim
import features as Ft
import cohort
import atlas as AT
import growth_trace as GT
import registry_v2 as R2
import run_intervention as RI
import run_steering as RS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_f")
TAG = "steering-2026-08-13"
N_MAT = 24
NMAX = 80
EPS = 1e-12


def state_metrics(n, beta, atl, cand, X9, gen_h, drift_rng_key):
    N = max(int(n.sum()), 1)
    x = n / N
    xp = x[n > 0]
    out_infl = float(x @ (x @ beta))          # weighted outgoing influence
    risk = float(R2.predict_v2(RI._BUNDLES[cand], X9[None, :],
                               Ft.graph_state_195(n, beta)[None, :])
                 ["v2"][0])
    d0 = AT.dist(n, atl)
    drifts = []
    for b in range(16):
        rb = cohort._rng(RI._ENT, 19, *drift_rng_key, b)
        br = sim.run_fissions(n, beta, cand, 1, rb)
        if br["n_done"]:
            drifts.append(AT.dist(br["daughters"][0], atl) - d0)
    return {
        "risk": risk, "rq": GT.r_q(n, beta), "atlas_dist": d0,
        "entropy": float(-np.sum(xp * np.log(xp + EPS))),
        "occupied": int((n > 0).sum()), "out_infl": out_infl,
        "gen_H": gen_h,
        "restoring_drift": float(np.mean(drifts)) if drifts else np.nan,
    }


def f2_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    cfg = "frozen02" if cand == "02" else "frozen03"
    atl = _ATLASES[(cfg, m)]
    recs = {"natural_composome": [], "written": [], "matched_natural": []}

    # probe lineage for natural composomes (domain 18)
    rng = cohort._rng(RI._ENT, 18, cand_i, m)
    probe = sim.run_fissions(n0, beta, cand, 120, rng)
    rqs = np.array([GT.r_q(d, beta) for d in probe["daughters"]])
    med = float(np.median(rqs))
    hits = [(i, d) for i, d in enumerate(probe["daughters"])
            if AT.nearest_sim(d, atl) >= 0.9 and rqs[i] > med]
    for i, d in hits[:2]:
        X9 = Ft.direct9(i + 1, 100, probe["H"][:i + 1], int(d.sum()))
        recs["natural_composome"].append(state_metrics(
            d, beta, atl, cand, X9, float(probe["H"][i]),
            (cand_i, m, 0, i)))

    # written state (Phase E write regeneration, rep 0)
    hs_w, holder = [], {}
    def wlog(f, n, swap, H, updates):
        hs_w.append(H)
        holder["n"] = n
    RS.steer_lineage(n0, beta, cand, cand_i, m, 0, "model_down", log=wlog)
    wn = holder["n"]
    X9w = Ft.direct9(len(hs_w), 100, np.array(hs_w), int(wn.sum()))
    recs["written"].append(state_metrics(
        wn, beta, atl, cand, X9w, float(hs_w[-1]), (cand_i, m, 1, 0)))

    # matched natural (noop lineage, fission 60)
    rn = cohort._rng(RI._ENT, 7, cand_i, m, 0)
    nat = sim.run_fissions(n0, beta, cand, 60, rn)
    nn = nat["daughters"][59]
    X9n = Ft.direct9(60, 100, nat["H"][:60], int(nn.sum()))
    recs["matched_natural"].append(state_metrics(
        nn, beta, atl, cand, X9n, float(nat["H"][59]), (cand_i, m, 2, 0)))
    return {"matrix": m, "candidate": cand, "recs": recs}


_ATLASES = None


def main():
    global _ATLASES
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)
    with open(os.path.join(OUT, "atlases.pkl"), "rb") as f:
        _ATLASES = pickle.load(f)

    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MAT)]
    with Pool(12) as pool:
        units = pool.map(f2_unit, jobs)
    print(f"F2 in {time.time()-t0:.0f}s")

    results = {}
    keys = ["risk", "rq", "atlas_dist", "entropy", "occupied",
            "out_infl", "gen_H", "restoring_drift"]
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        entry = {}
        for cls in ["natural_composome", "written", "matched_natural"]:
            rows = [r for u in cu for r in u["recs"][cls]]
            entry[cls] = {"n": len(rows)}
            for k in keys:
                v = np.array([r[k] for r in rows], dtype=float)
                entry[cls][k] = {"mean": float(np.nanmean(v)),
                                 "sd": float(np.nanstd(v))}
        results[cand] = entry
        print(f"\n=== F2 candidate {cand} ===")
        hdr = f"{'class':18s}" + "".join(f"{k:>12s}" for k in keys)
        print(hdr)
        for cls in ["natural_composome", "written", "matched_natural"]:
            row = f"{cls:18s}"
            for k in keys:
                row += f"{entry[cls][k]['mean']:12.3f}"
            print(row + f"   (n={entry[cls]['n']})")

    with open(os.path.join(OUT, "f2_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten:", os.path.join(OUT, "f2_results.json"))


if __name__ == "__main__":
    main()
