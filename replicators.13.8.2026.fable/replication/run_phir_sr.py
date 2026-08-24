"""Phase M: SR-state reading versus organizational reading
(PHIR_SR.md; sealed). Byte-exact replay of Phase J arms with full
per-fission recording; sliding 3-generation windows scored by four
instruments; SR labels from inherited-run lengths."""

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
import cohort
import phir_code as PC
import run_intervention as RI
from run_phir_bridge import traced_step, v2_scores
from run_phir_dose import ATOM_NAMES
import run_phir_confirm as J

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_phir_sr")
ARMS = ["ph_stab", "ph_destab", "random", "noop"]
L_GRID = [3, 5, 8]
L_PRIMARY = 5
BOOT_N = 4096
BOOT_SEED = 29

S_SRC = [ATOM_NAMES[a] for a in PC.ATOMS if a[0] == PC.S]
R_SRC = [ATOM_NAMES[a] for a in PC.ATOMS if a[0] == PC.R]
INSTRUMENTS = ["phiR", "printed", "synergy", "emergence"]


def window_scores(comps):
    """Four instruments from one pointwise PhiID on the macro halves
    of a window's update series. NaNs if the window is degenerate."""
    nan = {k: np.nan for k in INSTRUMENTS}
    if comps.shape[0] < 20:
        return nan
    Z = PC.clr(comps).T
    Z = Z[Z.std(axis=1) > PC.DEAD_EPS]
    if Z.shape[0] < 3:
        return nan
    Z = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(axis=1,
                                                    keepdims=True)
    a, b = PC.fiedler_bipartition(PC.mi_matrix_lag1(Z))
    if not a or not b:
        return nan
    edge = np.vstack([np.nanmean(Z[a], axis=0),
                      np.nanmean(Z[b], axis=0)])
    pi = PC.local_phi_id(edge)
    am = {ATOM_NAMES[at]: float(np.nanmean(pi[at])) for at in PC.ATOMS}
    syn = am[ATOM_NAMES[(PC.S, PC.S)]]
    caus = am[ATOM_NAMES[(PC.S, PC.U0)]] + am[ATOM_NAMES[(PC.S, PC.U1)]]
    return {"phiR": float(np.nanmean(PC.local_phi_r(pi))),
            "printed": sum(am[k] for k in S_SRC)
            - sum(am[k] for k in R_SRC),
            "synergy": syn, "emergence": syn + caus}


def sr_labels(hs, L):
    """Fission g (1-indexed) is SR iff inside a maximal inherited run
    of length >= L."""
    inh = np.asarray(hs) > sim.H_THRESH
    lab = np.zeros(len(inh), dtype=bool)
    i = 0
    while i < len(inh):
        if inh[i]:
            j = i
            while j < len(inh) and inh[j]:
                j += 1
            if j - i >= L:
                lab[i:j] = True
            i = j
        else:
            i += 1
    return lab


def unit(args):
    m, cand, rep, arm = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(J._r(0, m))
    n = sim.make_initial_state(J._r(1, m))
    rng = J._r(2, cand_i, m, rep)
    hs, record, marks = [], [], {}
    for f in range(1, J.STEER + 1):
        marks[f] = len(record)
        d, h = traced_step(n, beta, cand, rng, record)
        if d is None:
            break
        hs.append(h)
        n = d
        if f == J.STEER:
            break
        panel = J.draw_panel(n, J._r(3, cand_i, m, rep, f))
        if arm == "noop":
            continue
        if arm == "random":
            pick = panel[int(J._r(4, cand_i, m, rep, f)
                             .integers(J.PANEL))]
        else:
            sc = v2_scores(n, beta, cand, hs, f, panel)
            pick = panel[int(np.argmin(sc) if arm == "ph_stab"
                             else np.argmax(sc))]
        n = RI.apply_swap(n, pick)
    marks[len(hs) + 1] = len(record)
    comps = np.array(record, dtype=np.float64)
    nd = len(hs)
    code_late = np.nan
    if nd >= J.STEER and J.PHI_FROM in marks:
        code_late = float(PC.phi_r_code(comps[marks[J.PHI_FROM]:]))
    wins = []
    for g in range(2, nd):
        lo = marks.get(g - 1)
        hi = marks.get(g + 2, marks.get(nd + 1))
        if lo is None or hi is None or hi <= lo:
            continue
        sc = window_scores(comps[lo:hi])
        sc["g"] = g
        wins.append(sc)
    labs = {L: sr_labels(hs, L) for L in L_GRID}
    return {"matrix": m, "candidate": cand, "rep": rep, "arm": arm,
            "code_late": code_late, "n_done": nd,
            "wins": wins,
            "sr": {L: labs[L].tolist() for L in L_GRID},
            "sr_frac": {L: float(np.mean(labs[L])) for L in L_GRID}}


def delta_sr(u, inst, L):
    labs = np.array(u["sr"][L], dtype=bool)
    on, off = [], []
    for w in u["wins"]:
        v = w[inst]
        if not np.isfinite(v):
            continue
        (on if labs[w["g"] - 1] else off).append(v)
    if len(on) < 3 or len(off) < 3:
        return np.nan
    return float(np.mean(on) - np.mean(off))


def boot(vals_by_mat, n=BOOT_N, seed=BOOT_SEED):
    means = {m: np.nanmean(v) for m, v in vals_by_mat.items()
             if np.isfinite(v).any()}
    mats = list(means)
    if not mats:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [np.nanmean([means[mm] for mm in
                      rng.choice(mats, size=len(mats), replace=True)])
          for _ in range(n)]
    return (float(np.nanmean(list(means.values()))),
            float(np.nanquantile(bs, 0.025)),
            float(np.nanquantile(bs, 0.975)))


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = J._ENT

    t0 = time.time()
    jobs = [(m, c, r, a) for c in cohort.CANDIDATES
            for m in range(J.N_MAT) for r in J.REPS for a in ARMS]
    with Pool(12) as pool:
        units = pool.map(unit, jobs)
    print(f"Phase M replay in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "phir_sr_units.pkl"), "wb") as f:
        pickle.dump(units, f, protocol=4)

    with open(os.path.join(HERE, "results_phir_confirm",
                           "phir_confirm_units.pkl"), "rb") as f:
        stored = pickle.load(f)
    sk = {(u["matrix"], u["candidate"], u["rep"], u["arm"]):
          u["phi_code"] for u in stored}
    mism = sum(
        0 if (abs(u["code_late"] - sk[(u["matrix"], u["candidate"],
                                       u["rep"], u["arm"])]) < 1e-9
              or (np.isnan(u["code_late"])
                  and np.isnan(sk[(u["matrix"], u["candidate"],
                                   u["rep"], u["arm"])])))
        else 1 for u in units)
    print(f"REPLAY GATE: {'PASS' if mism == 0 else 'FAIL'} "
          f"({mism} mismatches)", flush=True)

    results = {"replay_gate": bool(mism == 0)}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        entry = {"sr_occupancy": {}, "M1": {}, "M2": {}, "M3": {},
                 "M1_secondary": {}}
        for arm in ARMS:
            au = [u for u in cu if u["arm"] == arm]
            entry["sr_occupancy"][arm] = {
                str(L): float(np.mean([u["sr_frac"][L] for u in au]))
                for L in L_GRID}
        for inst in INSTRUMENTS:
            per = {}
            for u in cu:
                if u["arm"] != "noop":
                    continue
                per.setdefault(u["matrix"], []).append(
                    delta_sr(u, inst, L_PRIMARY))
            d, lo, hi = boot({m: np.array(v) for m, v in per.items()})
            entry["M1"][inst] = {"dsr": d, "ci": [lo, hi],
                                 "excludes0": bool(lo > 0 or hi < 0)}
            for L in (3, 8):
                per = {}
                for u in cu:
                    if u["arm"] != "noop":
                        continue
                    per.setdefault(u["matrix"], []).append(
                        delta_sr(u, inst, L))
                d2, lo2, hi2 = boot({m: np.array(v)
                                     for m, v in per.items()})
                entry["M1_secondary"][f"{inst}_L{L}"] = \
                    {"dsr": d2, "ci": [lo2, hi2]}
            per = {}
            for u in cu:
                if u["arm"] not in ("ph_stab", "ph_destab"):
                    continue
                s = delta_sr(u, inst, L_PRIMARY)
                per.setdefault(u["matrix"], {}).setdefault(
                    u["arm"], []).append(s)
            diffs = {}
            for m, d3 in per.items():
                if "ph_stab" in d3 and "ph_destab" in d3:
                    a = np.nanmean(d3["ph_stab"])
                    b = np.nanmean(d3["ph_destab"])
                    if np.isfinite(a) and np.isfinite(b):
                        diffs[m] = np.array([a - b])
            d, lo, hi = boot(diffs)
            entry["M2"][inst] = {"diff": d, "ci": [lo, hi],
                                 "excludes0": bool(lo > 0 or hi < 0)}
            per = {}
            for u in cu:
                if u["arm"] not in ("ph_stab", "ph_destab"):
                    continue
                vals = [w[inst] for w in u["wins"]
                        if np.isfinite(w[inst])]
                if vals:
                    per.setdefault(u["matrix"], {}).setdefault(
                        u["arm"], []).append(np.mean(vals))
            diffs = {}
            for m, d3 in per.items():
                if "ph_stab" in d3 and "ph_destab" in d3:
                    diffs[m] = np.array([np.mean(d3["ph_stab"])
                                         - np.mean(d3["ph_destab"])])
            d, lo, hi = boot(diffs)
            entry["M3"][inst] = {"diff": d, "ci": [lo, hi],
                                 "excludes0": bool(lo > 0 or hi < 0)}
        results[cand] = entry

        print(f"\n=== Phase M candidate {cand} ===")
        print("SR occupancy (L=5): " + " ".join(
            f"{a}={entry['sr_occupancy'][a]['5']:.2f}" for a in ARMS))
        print(f"{'instrument':11s} {'M1 dSR(noop)':>22s} "
              f"{'M2 stab-destab dSR':>22s} {'M3 between-arm':>22s}")
        for inst in INSTRUMENTS:
            m1, m2, m3 = (entry["M1"][inst], entry["M2"][inst],
                          entry["M3"][inst])
            def fmt(v):
                return (f"{v['dsr'] if 'dsr' in v else v['diff']:+.3f}"
                        f" [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                        + ("*" if v["excludes0"] else " "))
            print(f"{inst:11s} {fmt(m1):>22s} {fmt(m2):>22s} "
                  f"{fmt(m3):>22s}")

    with open(os.path.join(OUT, "phir_sr_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "phir_sr_results.json"))


if __name__ == "__main__":
    main()
