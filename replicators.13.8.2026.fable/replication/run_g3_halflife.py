"""Phase G3: control half-life and minimum feedback rate
(preregistered in PHASE_G.md; domain 22).

Pulse ladder: model-down steering for P in {1,2,4,8,16,32,60} fissions
then 60 free fissions; anchor decay, inheritance, risk traces;
registered accumulation test (Spearman(P, post-release persistence)).
Periodic control: one model-down edit every {1,2,4,8,16} fissions,
budget-matched random arms.
Event-triggered: edit only when frozen risk > {0.15, 0.25, 0.35}.
Registered adjudication: continuous-required / sparse-sufficient /
accumulating-hysteresis. Registered prediction: sparse-sufficient,
no accumulation (half-life ~5-10 fissions regardless of pulse).
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
import run_intervention as RI
import run_steering as RS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_g")
TAG = "steering-2026-08-13"
N_MAT = 24
REPS = [0, 1]
PULSES = [1, 2, 4, 8, 16, 32, 60]
PERIODS = [1, 2, 4, 8, 16]
THRESH = [0.15, 0.25, 0.35]
FREE = 60


def risk_of(n, beta, cand, hs):
    X9 = Ft.direct9(max(len(hs), 1), 100, np.array(hs or [1.0]),
                    int(n.sum()))
    return float(RI.score_states(RI._BUNDLES[cand], X9,
                                 [Ft.graph_state_195(n, beta)])[0])


def steer_pulse(n0, beta, cand, cand_i, m, rep, pulse):
    """Model-down for `pulse` fissions (edits after fissions 1..pulse-1),
    then FREE for 60. Returns anchor trace, inherit, risk trace."""
    rng = cohort._rng(RI._ENT, 22, 0, cand_i, m, rep)
    n = n0.copy()
    hs = []
    for f in range(1, pulse + 1):
        step = sim.run_fissions(n, beta, cand, 1, rng)
        if step["n_done"] < 1:
            return None
        hs.append(float(step["H"][0]))
        n = step["final"]
        if f == pulse:
            break
        X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
        n = RI.apply_swap(n, RS.marginal_swap(n, beta, X9,
                                              RI._BUNDLES[cand], -1))
    anchor = n.astype(float)
    rel = sim.run_fissions(n, beta, cand, FREE, rng)
    d = rel["daughters"].astype(float)
    ah = np.array([sim.cosine_h(d[i], anchor) for i in range(len(d))])
    risks = [risk_of(rel["daughters"][i], beta, cand,
                     hs + list(rel["H"][:i + 1]))
             for i in range(0, len(d), 10)]
    return {"ah": ah, "inh": float(np.mean(rel["inherited"])),
            "risks": risks,
            "t07": int(np.argmax(ah < 0.7)) + 1 if (ah < 0.7).any()
            else FREE + 1,
            "ah10": float(ah[9]) if len(ah) >= 10 else np.nan}


def periodic(n0, beta, cand, cand_i, m, rep, period, random_arm):
    rng = cohort._rng(RI._ENT, 22, 1, cand_i, m, rep)
    n = n0.copy()
    hs, inh = [], []
    edits = 0
    for f in range(1, FREE + 1):
        step = sim.run_fissions(n, beta, cand, 1, rng)
        if step["n_done"] < 1:
            break
        hs.append(float(step["H"][0]))
        inh.append(step["inherited"][0])
        n = step["final"]
        if f == FREE or f % period != 0:
            continue
        if random_arm:
            rr = cohort._rng(RI._ENT, 22, 2, cand_i, m, rep, f)
            present = np.where(n > 0)[0]
            i = int(present[rr.integers(len(present))])
            j = int(rr.integers(sim.NG - 1))
            swap = (i, j + 1 if j >= i else j)
        else:
            X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
            swap = RS.marginal_swap(n, beta, X9, RI._BUNDLES[cand], -1)
        n = RI.apply_swap(n, swap)
        edits += 1
    return {"inh": float(np.mean(inh)) if inh else np.nan,
            "edits": edits,
            "final_risk": risk_of(n, beta, cand, hs)}


def event_triggered(n0, beta, cand, cand_i, m, rep, theta):
    rng = cohort._rng(RI._ENT, 22, 3, cand_i, m, rep)
    n = n0.copy()
    hs, inh = [], []
    edits = excursions = 0
    above = False
    for f in range(1, FREE + 1):
        step = sim.run_fissions(n, beta, cand, 1, rng)
        if step["n_done"] < 1:
            break
        hs.append(float(step["H"][0]))
        inh.append(step["inherited"][0])
        n = step["final"]
        if f == FREE:
            break
        q = risk_of(n, beta, cand, hs)
        if q > theta:
            if not above:
                excursions += 1
            above = True
            X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
            n = RI.apply_swap(n, RS.marginal_swap(
                n, beta, X9, RI._BUNDLES[cand], -1))
            edits += 1
        else:
            above = False
    return {"inh": float(np.mean(inh)) if inh else np.nan,
            "edits": edits, "excursions": excursions}


def unit(args):
    m, cand, rep = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    res = {"pulse": {}, "periodic": {}, "periodic_rand": {},
           "event": {}}
    for p in PULSES:
        r = steer_pulse(n0, beta, cand, cand_i, m, rep, p)
        if r is not None:
            res["pulse"][p] = {k: r[k] for k in
                               ("inh", "t07", "ah10", "risks")}
    for k in PERIODS:
        res["periodic"][k] = periodic(n0, beta, cand, cand_i, m, rep,
                                      k, False)
        res["periodic_rand"][k] = periodic(n0, beta, cand, cand_i, m,
                                           rep, k, True)
    for th in THRESH:
        res["event"][th] = event_triggered(n0, beta, cand, cand_i, m,
                                           rep, th)
    return {"matrix": m, "candidate": cand, "rep": rep, **res}


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)

    t0 = time.time()
    jobs = [(m, c, r) for c in cohort.CANDIDATES for m in range(N_MAT)
            for r in REPS]
    with Pool(12) as pool:
        units = pool.map(unit, jobs)
    print(f"G3 campaign in {time.time()-t0:.0f}s")
    with open(os.path.join(OUT, "g3_units.pkl"), "wb") as f:
        pickle.dump(units, f, protocol=4)

    results = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        mats = np.array([u["matrix"] for u in cu])
        entry = {"pulse": {}, "periodic": {}, "periodic_rand": {},
                 "event": {}}
        triples = []                      # (matrix, pulse, t07)
        for p in PULSES:
            rows = [(u["matrix"], u["pulse"][p]) for u in cu
                    if p in u["pulse"]]
            entry["pulse"][p] = {
                "half_life_t07": float(np.mean([r["t07"]
                                                for _, r in rows])),
                "anchor_at10": float(np.nanmean([r["ah10"]
                                                 for _, r in rows])),
                "release_inherit": float(np.mean([r["inh"]
                                                  for _, r in rows])),
            }
            triples.extend((mm, p, r["t07"]) for mm, r in rows)
        P = np.array([(p, t) for _, p, t in triples])
        rho = float(spearmanr(P[:, 0], P[:, 1]).correlation)
        boots = []
        rng = np.random.default_rng(5)
        um = np.unique(mats)
        per_mat = {mm: [(p, t) for m2, p, t in triples if m2 == mm]
                   for mm in um}
        for _ in range(1024):
            pick = rng.choice(um, size=len(um), replace=True)
            S = np.array([pair for mm in pick for pair in per_mat[mm]])
            boots.append(spearmanr(S[:, 0], S[:, 1]).correlation)
        ci = (float(np.nanquantile(boots, 0.025)),
              float(np.nanquantile(boots, 0.975)))
        entry["accumulation"] = {"spearman": rho, "ci": ci,
                                 "hysteresis": bool(rho > 0
                                                    and ci[0] > 0)}
        for k in PERIODS:
            entry["periodic"][k] = {
                "inherit": float(np.nanmean([u["periodic"][k]["inh"]
                                             for u in cu])),
                "edits": float(np.mean([u["periodic"][k]["edits"]
                                        for u in cu])),
                "final_risk": float(np.mean(
                    [u["periodic"][k]["final_risk"] for u in cu]))}
            entry["periodic_rand"][k] = {
                "inherit": float(np.nanmean(
                    [u["periodic_rand"][k]["inh"] for u in cu]))}
        for th in THRESH:
            entry["event"][th] = {
                "inherit": float(np.nanmean([u["event"][th]["inh"]
                                             for u in cu])),
                "edits": float(np.mean([u["event"][th]["edits"]
                                        for u in cu])),
                "excursions": float(np.mean(
                    [u["event"][th]["excursions"] for u in cu]))}
        p1 = entry["periodic"][1]["inherit"]
        sparse_ok = entry["periodic"][4]["inherit"] >= 0.95 \
            or entry["periodic"][8]["inherit"] >= 0.95
        continuous_req = entry["periodic"][2]["inherit"] < 0.95
        if entry["accumulation"]["hysteresis"]:
            verdict = "accumulating-hysteresis"
        elif sparse_ok:
            verdict = "sparse-sufficient"
        elif continuous_req:
            verdict = "continuous-correction-required"
        else:
            verdict = "intermediate"
        entry["verdict"] = verdict
        results[cand] = entry

        print(f"\n=== G3 candidate {cand} ===")
        print("pulse  half-life(t<0.7)  anchor@10  release-inherit")
        for p in PULSES:
            e = entry["pulse"][p]
            print(f"{p:5d}  {e['half_life_t07']:16.1f}  "
                  f"{e['anchor_at10']:9.2f}  {e['release_inherit']:15.3f}")
        print(f"accumulation Spearman {rho:+.3f} CI "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}] -> hysteresis "
              f"{entry['accumulation']['hysteresis']}")
        print("period  inherit  rand-inherit  edits  final-risk")
        for k in PERIODS:
            print(f"{k:6d}  {entry['periodic'][k]['inherit']:7.3f}  "
                  f"{entry['periodic_rand'][k]['inherit']:12.3f}  "
                  f"{entry['periodic'][k]['edits']:5.1f}  "
                  f"{entry['periodic'][k]['final_risk']:10.3f}")
        for th in THRESH:
            e = entry["event"][th]
            print(f"event q>{th}: inherit {e['inherit']:.3f} edits "
                  f"{e['edits']:.1f} excursions {e['excursions']:.1f}")
        print(f"VERDICT: {verdict}")

    with open(os.path.join(OUT, "g3_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "g3_results.json"))


if __name__ == "__main__":
    main()
