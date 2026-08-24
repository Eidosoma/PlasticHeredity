"""Phase F5b + F6 + negative control.

F5b — retire the ambiguous "lost" label: deterministically regenerate
the F3 challenge branches for t in {0, 60} x arms {none, k16} (same
spawn keys) capturing END compositions, and reclassify every branch
against the independent atlases:
    same_basin  : end atlas-sim >= 0.9 and same nearest center as start
    alt_basin   : end atlas-sim >= 0.9 and a DIFFERENT center
    unassigned  : end atlas-sim < 0.9
    extinct     : branch died
Tests the Kahana multistability reading of Phase E's "lost" outcomes.

F6 — moving/stochastic attractor: per (matrix, candidate), 4 pair-reps
of two lineages started from DISJOINT-support compositions, run 100
fissions under (a) common random streams (same seed; consumption
divergence caveat registered) and (b) independent streams. Registered
outcomes:
    pullback evidence      : mean last-20-fission cross-lineage
                             similarity, same-noise minus independent,
                             >= 0.10 with matrix-bootstrap CI > 0
    distributional evidence: paired lineages' composome-occupancy
                             histograms (last 40 fissions) closer than
                             a cross-matrix baseline (TV distance)
    neither                : process-without-destination support

Negative control — the (-4, 3) regime: within-growth atlas gain and
cross-generation drift measured as in F1/F4; registered expectation:
restoring signals much weaker or absent.
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
import run_intervention as RI
import run_steering as RS
import run_f3_f4 as F34

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_f")
TAG = "steering-2026-08-13"
N_MAT = 24
RECLS_T = [(0, 0), (60, 5)]          # (t, ti index in F34.DELAYS)
RECLS_ARMS = [("none", 0), ("k16", 2)]
N_BR = 32
REC = 24
N_PAIR = 4
N_FISS_F6 = 100
NEG_TAG = "knob-A-4_S3-2026-08-13"
N_NEG_MAT = 8

_ATLASES = None


def f5b_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    cfg = "frozen02" if cand == "02" else "frozen03"
    atl = _ATLASES[(cfg, m)]

    hs_w, holder = [], {}
    def wlog(f, n, swap, H, updates):
        hs_w.append(H)
        holder["n"] = n
    RS.steer_lineage(n0, beta, cand, cand_i, m, 0, "model_down", log=wlog)
    written = holder["n"]
    rr = cohort._rng(RI._ENT, 9, cand_i, m, 0)
    rel = sim.run_fissions(written, beta, cand, 60, rr)
    states_at = {0: written, 60: rel["daughters"][59]
                 if rel["n_done"] >= 60 else rel["final"]}

    out = {}
    for (t, ti) in RECLS_T:
        s = states_at[t]
        hs_full = hs_w + list(rel["H"][:t])
        X9 = Ft.direct9(len(hs_full), 100, np.array(hs_full),
                        int(s.sum()))
        start_c = AT.nearest_center(s, atl)
        for (arm, ai) in RECLS_ARMS:
            pr = cohort._rng(RI._ENT, 11, cand_i, m, 0, ti, ai)
            s0 = F34.apply_perturbation(arm, s, beta, cand, atl, pr, X9)
            counts = {"same_basin": 0, "alt_basin": 0,
                      "unassigned": 0, "extinct": 0}
            for b in range(N_BR):
                rb = cohort._rng(RI._ENT, 13, cand_i, m, 0, ti, ai, b)
                br = sim.run_fissions(s0, beta, cand, REC, rb)
                if br["n_done"] < REC:
                    counts["extinct"] += 1
                    continue
                end = br["daughters"][-1]
                simv = AT.nearest_sim(end, atl)
                if simv < 0.9:
                    counts["unassigned"] += 1
                elif AT.nearest_center(end, atl) == start_c:
                    counts["same_basin"] += 1
                else:
                    counts["alt_basin"] += 1
            out[f"{t}_{arm}"] = counts
    return {"matrix": m, "candidate": cand, "counts": out,
            "atlas_k": _ATLASES[(cfg, m)]["k"]}


def _disjoint_inits(ent, cand_i, m, rep):
    r1 = cohort._rng(ent, 15, cand_i, m, rep, 0)
    types_a = r1.choice(sim.NG, size=40, replace=False)
    rest = np.setdiff1d(np.arange(sim.NG), types_a)
    r2 = cohort._rng(ent, 15, cand_i, m, rep, 1)
    types_b = rest[r2.choice(len(rest), size=40, replace=False)]
    na = np.zeros(sim.NG, dtype=np.int64)
    nb = np.zeros(sim.NG, dtype=np.int64)
    na[types_a] = 1
    nb[types_b] = 1
    return na, nb


def f6_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, _ = cohort.matrix_and_init(RI._ENT, m)
    cfg = "frozen02" if cand == "02" else "frozen03"
    atl = _ATLASES[(cfg, m)]
    res = {"same": [], "indep": [], "occ": []}
    for rep in range(N_PAIR):
        na, nb = _disjoint_inits(RI._ENT, cand_i, m, rep)
        for cond in ("same", "indep"):
            if cond == "same":
                ka = kb = (15, cand_i, m, rep, 7)
            else:
                ka = (15, cand_i, m, rep, 8)
                kb = (15, cand_i, m, rep, 9)
            ta = sim.run_fissions(na, beta, cand, N_FISS_F6,
                                  cohort._rng(RI._ENT, *ka))
            tb = sim.run_fissions(nb, beta, cand, N_FISS_F6,
                                  cohort._rng(RI._ENT, *kb))
            L = min(ta["n_done"], tb["n_done"])
            if L < 40:
                continue
            hab = [sim.cosine_h(ta["daughters"][i].astype(float),
                                tb["daughters"][i].astype(float))
                   for i in range(L - 20, L)]
            res[cond].append(float(np.mean(hab)))
            if cond == "indep":
                k = atl["k"]
                ha = np.bincount(
                    [AT.nearest_center(d, atl)
                     for d in ta["daughters"][L - 40:L]],
                    minlength=k) / 40
                hb = np.bincount(
                    [AT.nearest_center(d, atl)
                     for d in tb["daughters"][L - 40:L]],
                    minlength=k) / 40
                res["occ"].append(
                    (ha.tolist(), hb.tolist(),
                     float(0.5 * np.abs(ha - hb).sum())))
    return {"matrix": m, "candidate": cand, **res}


def neg_unit(m):
    ent = cohort.domain_entropy("confirmation", NEG_TAG)
    beta = sim.make_beta(cohort._rng(ent, 0, m), a_mu=-4.0, sigma=3.0)
    n0 = sim.make_initial_state(cohort._rng(ent, 1, m))
    # small atlas from 2 lineages
    pres = []
    for li in range(2):
        rng = cohort._rng(ent, 12, 0, m, li)
        o = GT.traced_run_fissions(n0, beta, "02", 300, rng, 80,
                                   grid_step=80)
        pres.extend(r["parent"].astype(float) for r in o["recs"])
    P = np.array(pres)
    keep = [0] + [i for i in range(1, len(P))
                  if sim.cosine_h(P[i], P[i - 1]) > sim.H_THRESH]
    Pk = P[keep] if len(keep) >= 10 else P
    C, lab = AT._kmeans_cosine(Pk[:300], min(3, len(Pk)), seed=m)
    atl = {"centers": C, "k": len(C)}
    wg, cg = [], []
    for ic in range(6):
        rng = cohort._rng(ent, 17, 0, m, ic)
        nn = np.bincount(rng.integers(0, sim.NG, 40),
                         minlength=sim.NG).astype(np.int64)
        o = GT.traced_run_fissions(nn, beta, "02", 20, rng, 80,
                                   grid_step=5)
        dprev = None
        for r in o["recs"][4:]:
            sm, sc = r["snaps"][0]
            em, ec = r["snaps"][-1]
            wg.append(AT.nearest_sim(ec, atl) - AT.nearest_sim(sc, atl))
            d = 1 - AT.nearest_sim(r["daughter"], atl)
            if dprev is not None:
                cg.append(d - dprev)
            dprev = d
    return {"matrix": m, "wg": float(np.mean(wg)),
            "cg": float(np.mean(cg)), "nondrift_frac":
            float(len(keep) / len(P))}


def main():
    global _ATLASES
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)
    with open(os.path.join(OUT, "atlases.pkl"), "rb") as f:
        _ATLASES = pickle.load(f)
    F34._ATLASES = _ATLASES

    results = {}

    # ---------------- F5b ---------------------------------------------
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MAT)]
    with Pool(12) as pool:
        units = pool.map(f5b_unit, jobs)
    print(f"F5b in {time.time()-t0:.0f}s")
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        entry = {}
        multi = [u for u in cu if u["atlas_k"] > 1]
        for key in [f"{t}_{a}" for t, _ in RECLS_T for a, _ in RECLS_ARMS]:
            tot = np.array([sum(u["counts"][key].values()) for u in cu])
            agg = {c: float(np.sum([u["counts"][key][c] for u in cu])
                            / tot.sum())
                   for c in ("same_basin", "alt_basin", "unassigned",
                             "extinct")}
            agg_multi = {c: float(np.sum([u["counts"][key][c]
                                          for u in multi])
                         / max(np.sum([sum(u["counts"][key].values())
                                       for u in multi]), 1))
                         for c in ("same_basin", "alt_basin")}
            entry[key] = {"all": agg, "multi_atlas_only": agg_multi}
        results[f"f5b_{cand}"] = entry
        print(f"\n=== F5b candidate {cand} (fraction of branches) ===")
        for key, v in entry.items():
            a = v["all"]
            print(f"{key:10s} same {a['same_basin']:.2f} | alt "
                  f"{a['alt_basin']:.2f} | unassigned "
                  f"{a['unassigned']:.2f} | extinct {a['extinct']:.2f}"
                  f"  (multi-composome matrices: same "
                  f"{v['multi_atlas_only']['same_basin']:.2f} alt "
                  f"{v['multi_atlas_only']['alt_basin']:.2f})")

    # ---------------- F6 ----------------------------------------------
    t0 = time.time()
    with Pool(12) as pool:
        f6 = pool.map(f6_unit, jobs)
    print(f"\nF6 in {time.time()-t0:.0f}s")
    for cand in cohort.CANDIDATES:
        cu = [u for u in f6 if u["candidate"] == cand]
        same = np.concatenate([u["same"] for u in cu if u["same"]])
        indep = np.concatenate([u["indep"] for u in cu if u["indep"]])
        mats_s = np.concatenate([[u["matrix"]] * len(u["same"])
                                 for u in cu if u["same"]])
        diff = []
        for u in cu:
            k = min(len(u["same"]), len(u["indep"]))
            diff.extend(np.array(u["same"][:k])
                        - np.array(u["indep"][:k]))
        diff = np.array(diff)
        mats_d = np.concatenate([[u["matrix"]]
                                 * min(len(u["same"]), len(u["indep"]))
                                 for u in cu])
        rng = np.random.default_rng(31415)
        ci = RI.boot_lower(diff, mats_d, rng, n=1024)
        tv_paired = [o[2] for u in cu for o in u["occ"]]
        # cross-matrix baseline: occupancy of matrix i vs matrix j
        occs = [(u["matrix"], o[0]) for u in cu for o in u["occ"]]
        tv_cross = []
        rngx = np.random.default_rng(7)
        for _ in range(200):
            i, j = rngx.integers(len(occs)), rngx.integers(len(occs))
            if occs[i][0] == occs[j][0]:
                continue
            k = max(len(occs[i][1]), len(occs[j][1]))
            a = np.pad(occs[i][1], (0, k - len(occs[i][1])))
            b = np.pad(occs[j][1], (0, k - len(occs[j][1])))
            tv_cross.append(float(0.5 * np.abs(a - b).sum()))
        entry = {
            "same_noise_H": float(np.mean(same)),
            "indep_noise_H": float(np.mean(indep)),
            "pullback_diff": {"mean": float(diff.mean()),
                              "ci": ci},
            "pullback_pass": bool(diff.mean() >= 0.10 and ci[0] > 0),
            "tv_paired_indep": float(np.mean(tv_paired)),
            "tv_cross_matrix": float(np.mean(tv_cross)),
        }
        results[f"f6_{cand}"] = entry
        print(f"F6 cand {cand}: same-noise H {entry['same_noise_H']:.3f}"
              f" vs indep {entry['indep_noise_H']:.3f} | diff "
              f"{diff.mean():+.3f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] | "
              f"pullback margin pass: {entry['pullback_pass']}")
        print(f"  occupancy TV: paired-indep "
              f"{entry['tv_paired_indep']:.3f} vs cross-matrix "
              f"{entry['tv_cross_matrix']:.3f}")

    # ---------------- negative control (-4, 3) ------------------------
    t0 = time.time()
    with Pool(8) as pool:
        neg = pool.map(neg_unit, range(N_NEG_MAT))
    print(f"\nnegative control in {time.time()-t0:.0f}s")
    results["neg_control"] = {
        "within_growth_gain": float(np.mean([n["wg"] for n in neg])),
        "cross_gen_drift": float(np.mean([n["cg"] for n in neg])),
        "nondrift_frac": float(np.mean([n["nondrift_frac"]
                                        for n in neg])),
    }
    nc = results["neg_control"]
    print(f"sigma=3 regime: within-growth gain {nc['within_growth_gain']:+.4f}"
          f" (home +0.033) | cross-gen drift {nc['cross_gen_drift']:+.4f}"
          f" | nondrift fraction {nc['nondrift_frac']:.2f}")

    with open(os.path.join(OUT, "f5b_f6_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "f5b_f6_results.json"))


if __name__ == "__main__":
    main()
