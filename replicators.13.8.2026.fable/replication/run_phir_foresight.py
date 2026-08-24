"""Phase N: the foresight round (PHIR_FORESIGHT.md; sealed).
N1 powered natural prediction; N2 event-locked early warning;
N3 generational-clock gauge check via Phase J replay."""

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
import phir_code as PC
import run_intervention as RI
from run_phir_bridge import traced_step, v2_scores
from run_phir_sr import window_scores, INSTRUMENTS
import run_phir_confirm as J

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_phir_foresight")
TAG = "phir-foresight-2026-08-18"
DOMAIN = 30
N_MAT = 16
N_REP = 12
STEER = 60
BOOT_N = 4096
BOOT_SEED = 31

_ENT = cohort.domain_entropy("confirmation", TAG)


def _r(*key):
    return cohort._rng(_ENT, DOMAIN, *key)


# ---- N1/N2 unit ------------------------------------------------------

def n1_unit(args):
    m, cand, rep = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(_r(0, m))
    n = sim.make_initial_state(_r(1, m))
    rng = _r(2, cand_i, m, rep)
    hs, record, marks = [], [], {}
    daughters = []
    for f in range(1, STEER + 1):
        marks[f] = len(record)
        d, h = traced_step(n, beta, cand, rng, record)
        if d is None:
            break
        hs.append(h)
        daughters.append(d.copy())
        n = d
    marks[len(hs) + 1] = len(record)
    nd = len(hs)
    if nd < STEER:
        return None
    comps = np.array(record, dtype=np.float64)
    D = np.array(daughters, dtype=np.float64)

    # per-window instruments over all centers (N2 + volatility)
    wins = {}
    for g in range(2, nd):
        lo, hi = marks[g - 1], marks.get(g + 2, marks[nd + 1])
        if hi > lo:
            wins[g] = window_scores(comps[lo:hi])

    # N1 predictors (fissions <= 40 only)
    w2140 = comps[marks[21]:marks[41]]
    sc = window_scores(w2140)
    phiR_series = [wins[g]["phiR"] for g in range(22, 40)
                   if g in wins and np.isfinite(wins[g]["phiR"])]
    vol = float(np.std(phiR_series)) if len(phiR_series) > 5 else np.nan
    if len(phiR_series) > 5:
        xs = np.arange(len(phiR_series))
        trend = float(np.polyfit(xs, phiR_series, 1)[0])
    else:
        trend = np.nan
    gen = window_scores(D[:40])
    state40 = daughters[39]
    X9 = Ft.direct9(40, 100, np.array(hs[:40]), int(state40.sum()))
    v2 = float(RI.score_states(RI._BUNDLES[cand], X9,
                               [Ft.graph_state_195(state40, beta)])[0])
    inh = np.array(hs) > sim.H_THRESH
    preds = {"phiR_scalar": sc["phiR"], "causation":
             sc["emergence"] - sc["synergy"], "emergence":
             sc["emergence"], "printed": sc["printed"],
             "phi_volatility": vol, "phi_trend": trend,
             "gen_phiR": gen["phiR"], "gen_printed": gen["printed"],
             "v2_risk": v2, "hist": float(np.mean(inh[20:40]))}
    out = {"matrix": m, "candidate": cand, "rep": rep,
           "breaks_late": int((~inh[40:60]).sum()), **preds}

    # N2 events: break at t with >= 5 inherited before it
    run = 0
    runs_before = []
    for v in inh:
        runs_before.append(run)
        run = run + 1 if v else 0
    cases_g, controls_g = [], []
    for t in range(1, nd + 1):
        g = t - 2
        if g not in wins:
            continue
        if not inh[t - 1] and runs_before[t - 1] >= 5:
            cases_g.append(g)
        elif (inh[t - 1] and runs_before[t - 1] >= 5
              and all(inh[t - 1 + k] for k in (1, 2)
                      if t - 1 + k < nd)):
            controls_g.append(g)
    def _vol(gs):
        vv = []
        for g in gs:
            prev = [wins[g - k]["phiR"] for k in (1, 2, 3)
                    if g - k in wins
                    and np.isfinite(wins[g - k]["phiR"])]
            if len(prev) == 3:
                vv.append(float(np.std(prev)))
        return vv
    volc = {"case": _vol(cases_g), "control": _vol(controls_g)}
    out["n2"] = {
        "n_case": len(cases_g), "n_control": len(controls_g),
        **{f"case_{k}": float(np.nanmean([wins[g][k]
                                          for g in cases_g]))
           if cases_g else np.nan for k in INSTRUMENTS},
        **{f"control_{k}": float(np.nanmean([wins[g][k]
                                             for g in controls_g]))
           if controls_g else np.nan for k in INSTRUMENTS},
        "case_vol": float(np.nanmean(volc["case"]))
        if volc["case"] else np.nan,
        "control_vol": float(np.nanmean(volc["control"]))
        if volc["control"] else np.nan}
    return out


# ---- N3 unit (Phase J replay, daughters only) ------------------------

def n3_unit(args):
    m, cand, rep, arm = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(J._r(0, m))
    n = sim.make_initial_state(J._r(1, m))
    rng = J._r(2, cand_i, m, rep)
    hs, daughters = [], []
    for f in range(1, J.STEER + 1):
        step = sim.run_fissions(n, beta, cand, 1, rng)
        if step["n_done"] < 1:
            break
        hs.append(float(step["H"][0]))
        n = step["final"]
        daughters.append(n.copy())
        if f == J.STEER:
            break
        panel = J.draw_panel(n, J._r(3, cand_i, m, rep, f))
        if arm == "noop":
            continue
        sc = v2_scores(n, beta, cand, hs, f, panel)
        pick = panel[int(np.argmin(sc) if arm == "ph_stab"
                         else np.argmax(sc))]
        n = RI.apply_swap(n, pick)
    inh = np.array(hs) > sim.H_THRESH
    gen = window_scores(np.array(daughters, dtype=np.float64))
    return {"matrix": m, "candidate": cand, "rep": rep, "arm": arm,
            "inherit": float(inh.mean()) if len(inh) else np.nan,
            "gen_phiR": gen["phiR"], "gen_printed": gen["printed"]}


# ---- stats -----------------------------------------------------------

def spearman_cells(rows, xkey, ykey, centered, seed=BOOT_SEED):
    if centered:
        mx = {}
        for r in rows:
            mx.setdefault(r["matrix"], []).append(r)
        rows2 = []
        for m, rs in mx.items():
            xb = np.nanmean([r[xkey] for r in rs])
            yb = np.nanmean([r[ykey] for r in rs])
            rows2 += [{"matrix": m, xkey: r[xkey] - xb,
                       ykey: r[ykey] - yb} for r in rs]
        rows = rows2
    per = {}
    for r in rows:
        if np.isfinite(r[xkey]) and np.isfinite(r[ykey]):
            per.setdefault(r["matrix"], []).append((r[xkey], r[ykey]))

    def fn(groups):
        xs = [x for g in groups for x, _ in g]
        ys = [y for g in groups for _, y in g]
        if len(xs) < 6 or np.std(xs) == 0 or np.std(ys) == 0:
            return np.nan
        return spearmanr(xs, ys).correlation
    mats = sorted(per)
    point = fn([per[m] for m in mats])
    rng = np.random.default_rng(seed)
    bs = [fn([per[m] for m in rng.choice(mats, len(mats), True)])
          for _ in range(BOOT_N)]
    return (float(point), float(np.nanquantile(bs, 0.025)),
            float(np.nanquantile(bs, 0.975)))


def mat_boot_vals(per_mat, seed=BOOT_SEED):
    means = {m: np.nanmean(v) for m, v in per_mat.items()
             if np.isfinite(v).any()}
    mats = list(means)
    if not mats:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [np.nanmean([means[mm] for mm in
                      rng.choice(mats, len(mats), True)])
          for _ in range(BOOT_N)]
    return (float(np.nanmean(list(means.values()))),
            float(np.nanquantile(bs, 0.025)),
            float(np.nanquantile(bs, 0.975)))


PREDICTORS = ["phiR_scalar", "causation", "emergence", "printed",
              "phi_volatility", "phi_trend", "gen_phiR",
              "gen_printed", "v2_risk", "hist"]


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = _ENT

    t0 = time.time()
    jobs = [(m, c, r) for c in cohort.CANDIDATES
            for m in range(N_MAT) for r in range(N_REP)]
    with Pool(12) as pool:
        n1 = [u for u in pool.map(n1_unit, jobs) if u]
    print(f"N1/N2 cohort in {time.time()-t0:.0f}s "
          f"({len(n1)} complete lineages)", flush=True)

    t0 = time.time()
    jobs = [(m, c, r, a) for c in cohort.CANDIDATES
            for m in range(J.N_MAT) for r in J.REPS
            for a in ("ph_stab", "ph_destab", "noop")]
    with Pool(12) as pool:
        n3 = pool.map(n3_unit, jobs)
    print(f"N3 replay in {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(HERE, "results_phir_confirm",
                           "phir_confirm_units.pkl"), "rb") as f:
        stored = pickle.load(f)
    sk = {(u["matrix"], u["candidate"], u["rep"], u["arm"]):
          u["inherit"] for u in stored}
    mism = sum(0 if u["inherit"] == sk[(u["matrix"], u["candidate"],
                                        u["rep"], u["arm"])] else 1
               for u in n3)
    print(f"N3 REPLAY GATE: {'PASS' if mism == 0 else 'FAIL'} "
          f"({mism} mismatches)", flush=True)

    with open(os.path.join(OUT, "foresight_units.pkl"), "wb") as f:
        pickle.dump({"n1": [{k: v for k, v in u.items()}
                            for u in n1], "n3": n3}, f, protocol=4)

    results = {"n3_replay_gate": bool(mism == 0)}
    for cand in cohort.CANDIDATES:
        cu = [u for u in n1 if u["candidate"] == cand]
        entry = {"N1": {}, "N1_residual": {}, "N2": {}, "N3": {}}
        # N1
        for p in PREDICTORS:
            for cen in (False, True):
                r, lo, hi = spearman_cells(cu, p, "breaks_late", cen)
                entry["N1"][f"{p}_{'centered' if cen else 'overall'}"] = \
                    {"rho": r, "ci": [lo, hi],
                     "excludes0": bool(lo > 0 or hi < 0)}
        # residual-on-v2 (centered), secondary
        mx = {}
        for u in cu:
            mx.setdefault(u["matrix"], []).append(u)
        rows = []
        for m, rs in mx.items():
            v2c = np.array([u["v2_risk"] for u in rs], dtype=float)
            yc = np.array([u["breaks_late"] for u in rs], dtype=float)
            v2c -= np.nanmean(v2c)
            yc -= np.nanmean(yc)
            b = (np.nansum(v2c * yc)
                 / max(np.nansum(v2c ** 2), 1e-12))
            for u, res in zip(rs, yc - b * v2c):
                rows.append({**u, "resid": float(res)})
        for p in ("phiR_scalar", "causation", "emergence",
                  "phi_volatility", "gen_phiR"):
            r, lo, hi = spearman_cells(rows, p, "resid", True)
            entry["N1_residual"][p] = {"rho": r, "ci": [lo, hi],
                                       "excludes0": bool(lo > 0
                                                         or hi < 0)}
        # N2
        for k in INSTRUMENTS + ["vol"]:
            per = {}
            for u in cu:
                a = u["n2"].get(f"case_{k}" if k != "vol"
                                else "case_vol")
                b = u["n2"].get(f"control_{k}" if k != "vol"
                                else "control_vol")
                if a is not None and b is not None \
                        and np.isfinite(a) and np.isfinite(b):
                    per.setdefault(u["matrix"], []).append(a - b)
            d, lo, hi = mat_boot_vals(
                {m: np.array(v) for m, v in per.items()})
            entry["N2"][k] = {"diff": d, "ci": [lo, hi],
                              "excludes0": bool(lo > 0 or hi < 0)}
        entry["N2"]["n_events"] = int(np.sum(
            [u["n2"]["n_case"] for u in cu]))
        # N3
        c3 = [u for u in n3 if u["candidate"] == cand]
        for key in ("gen_phiR", "gen_printed", "inherit"):
            per = {}
            for u in c3:
                if u["arm"] not in ("ph_stab", "ph_destab"):
                    continue
                if np.isfinite(u[key]):
                    per.setdefault(u["matrix"], {}).setdefault(
                        u["arm"], []).append(u[key])
            diffs = {}
            for m, d3 in per.items():
                if "ph_stab" in d3 and "ph_destab" in d3:
                    diffs[m] = np.array([np.mean(d3["ph_stab"])
                                         - np.mean(d3["ph_destab"])])
            d, lo, hi = mat_boot_vals(diffs)
            entry["N3"][key] = {"diff": d, "ci": [lo, hi],
                                "excludes0": bool(lo > 0 or hi < 0)}
        results[cand] = entry

        print(f"\n=== Phase N candidate {cand} "
              f"({len(cu)} lineages, {entry['N2']['n_events']} "
              f"pre-break events) ===")
        print("N1 (Spearman vs later breaks): overall | centered")
        for p in PREDICTORS:
            o = entry["N1"][f"{p}_overall"]
            c = entry["N1"][f"{p}_centered"]
            print(f"  {p:14s} {o['rho']:+.3f} "
                  f"[{o['ci'][0]:+.3f},{o['ci'][1]:+.3f}]"
                  f"{'*' if o['excludes0'] else ' '} | "
                  f"{c['rho']:+.3f} "
                  f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]"
                  f"{'*' if c['excludes0'] else ' '}")
        print("N1 residual-on-v2 (centered):")
        for p, v in entry["N1_residual"].items():
            print(f"  {p:14s} {v['rho']:+.3f} "
                  f"[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                  + ("*" if v["excludes0"] else ""))
        print("N2 (pre-break minus deep-run control):")
        for k in INSTRUMENTS + ["vol"]:
            v = entry["N2"][k]
            print(f"  {k:10s} {v['diff']:+.4f} "
                  f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
                  + ("*" if v["excludes0"] else ""))
        print("N3 (gen-clock, ph_stab - ph_destab):")
        for k, v in entry["N3"].items():
            print(f"  {k:12s} {v['diff']:+.4f} "
                  f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
                  + ("*" if v["excludes0"] else ""))

    with open(os.path.join(OUT, "foresight_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "foresight_results.json"))


if __name__ == "__main__":
    main()
