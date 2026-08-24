"""Phase K: Phi_R dose-response (K1), atom decomposition of Phase J
(K2), natural-prediction test (K3), instrument robustness (K4).
Preregistered in PHIR_DOSE.md; sealed before the campaign."""

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
import run_phir_confirm as J

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_phir_dose")
TAG = "phir-dose-2026-08-17"
DOMAIN = 29
N_MAT = 24
N_MAT_NAT = 48
REPS = [0, 1]
STEER = 60
PHI_FROM = 41
PANEL = 12
CADENCES = [1, 2, 4, 8, 16]
ARMS = [f"{d}{k}" for d in ("stab", "destab") for k in CADENCES] \
    + ["noop"]
BOOT_N = 4096
BOOT_SEED = 19

_ENT = cohort.domain_entropy("confirmation", TAG)


def _r(*key):
    return cohort._rng(_ENT, DOMAIN, *key)


# ---- atom-level Phi (reuses the sealed phir_code internals) ----------

ATOM_NAMES = {a: f"{'+'.join(str(p) for p in a[0])}->"
              f"{'+'.join(str(p) for p in a[1])}" for a in PC.ATOMS}


def phi_atoms(comps):
    """Mean of each pointwise atom over the window, plus the scalar
    Phi_R and the authors' alternative summaries. Mirrors
    phir_code.phi_r_code exactly up to the final aggregation."""
    comps = np.asarray(comps, dtype=np.float64)
    if comps.shape[0] < 20:
        return None
    Z = PC.clr(comps).T
    Z = Z[Z.std(axis=1) > PC.DEAD_EPS]
    if Z.shape[0] < 3:
        return None
    Z = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(axis=1,
                                                    keepdims=True)
    a, b = PC.fiedler_bipartition(PC.mi_matrix_lag1(Z))
    if not a or not b:
        return None
    edge = np.vstack([np.nanmean(Z[a], axis=0),
                      np.nanmean(Z[b], axis=0)])
    pi = PC.local_phi_id(edge)
    out = {ATOM_NAMES[at]: float(np.nanmean(pi[at]))
           for at in PC.ATOMS}
    out["phi_r"] = float(np.nanmean(PC.local_phi_r(pi)))
    sts = (PC.S, PC.S)
    out["synergy"] = out[ATOM_NAMES[sts]]
    out["causation"] = out[ATOM_NAMES[(PC.S, PC.U0)]] \
        + out[ATOM_NAMES[(PC.S, PC.U1)]]
    out["emergence"] = out["synergy"] + out["causation"]
    # K4 variants (computed only when requested by caller subsample)
    return out


def phi_variants(comps):
    """K4: three one-knob instrument variants."""
    comps = np.asarray(comps, dtype=np.float64)
    res = {}
    # (i) no CLR
    Z = comps.T
    Z = Z[Z.std(axis=1) > PC.DEAD_EPS]
    if Z.shape[0] >= 3 and comps.shape[0] >= 20:
        Z = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(
            axis=1, keepdims=True)
        a, b = PC.fiedler_bipartition(PC.mi_matrix_lag1(Z))
        if a and b:
            edge = np.vstack([np.nanmean(Z[a], axis=0),
                              np.nanmean(Z[b], axis=0)])
            res["no_clr"] = float(np.nanmean(
                PC.local_phi_r(PC.local_phi_id(edge))))
    # (ii) median aggregation and (iii) drop-last CLR component
    for name, drop in (("median_agg", False), ("drop_last", True)):
        Z = PC.clr(comps)
        if drop:
            Z = Z[:, :-1]
        Z = Z.T
        Z = Z[Z.std(axis=1) > PC.DEAD_EPS]
        if Z.shape[0] < 3 or comps.shape[0] < 20:
            continue
        Z = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(
            axis=1, keepdims=True)
        a, b = PC.fiedler_bipartition(PC.mi_matrix_lag1(Z))
        if not (a and b):
            continue
        edge = np.vstack([np.nanmean(Z[a], axis=0),
                          np.nanmean(Z[b], axis=0)])
        v = PC.local_phi_r(PC.local_phi_id(edge))
        res[name] = float(np.nanmedian(v) if name == "median_agg"
                          else np.nanmean(v))
    return res


# ---- K1: dose campaign ----------------------------------------------

def k1_unit(args):
    m, cand, rep, arm = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(_r(0, m))
    n = sim.make_initial_state(_r(1, m))
    rng = _r(2, cand_i, m, rep)
    if arm == "noop":
        direction, k = None, None
    else:
        direction = "stab" if arm.startswith("stab") else "destab"
        k = int(arm[len(direction):])
    hs, record = [], []
    edits = 0
    for f in range(1, STEER + 1):
        if f >= PHI_FROM:
            d, h = traced_step(n, beta, cand, rng, record)
            if d is None:
                break
            hs.append(h)
            n = d
        else:
            step = sim.run_fissions(n, beta, cand, 1, rng)
            if step["n_done"] < 1:
                break
            hs.append(float(step["H"][0]))
            n = step["final"]
        if f == STEER:
            break
        panel = J.draw_panel(n, _r(3, cand_i, m, rep, f))
        if direction is None or f % k != 0:
            continue
        sc = v2_scores(n, beta, cand, hs, f, panel)
        pick = panel[int(np.argmin(sc) if direction == "stab"
                         else np.argmax(sc))]
        n = RI.apply_swap(n, pick)
        edits += 1
    hs = np.array(hs)
    inh = hs > sim.H_THRESH
    comps = np.array(record, dtype=np.float64) if record else None
    sign = 0 if direction is None else (1 if direction == "stab"
                                        else -1)
    return {"matrix": m, "candidate": cand, "rep": rep, "arm": arm,
            "dose": sign * edits / 59.0,
            "inherit": float(inh.mean()) if len(inh) else np.nan,
            "phi": float(PC.phi_r_code(comps))
            if comps is not None else np.nan}


# ---- K2/K4: Phase J replay with atoms -------------------------------

def k2_unit(args):
    m, cand, rep, arm = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(J._r(0, m))
    n = sim.make_initial_state(J._r(1, m))
    rng = J._r(2, cand_i, m, rep)
    hs, record = [], []
    for f in range(1, J.STEER + 1):
        if f >= J.PHI_FROM:
            d, h = traced_step(n, beta, cand, rng, record)
            if d is None:
                break
            hs.append(h)
            n = d
        else:
            step = sim.run_fissions(n, beta, cand, 1, rng)
            if step["n_done"] < 1:
                break
            hs.append(float(step["H"][0]))
            n = step["final"]
        if f == J.STEER:
            break
        panel = J.draw_panel(n, J._r(3, cand_i, m, rep, f))
        if arm == "noop":
            continue
        sc = v2_scores(n, beta, cand, hs, f, panel)
        pick = panel[int(np.argmin(sc) if arm == "ph_stab"
                         else np.argmax(sc))]
        n = RI.apply_swap(n, pick)
    comps = np.array(record, dtype=np.float64) if record else None
    atoms = phi_atoms(comps) if comps is not None else None
    out = {"matrix": m, "candidate": cand, "rep": rep, "arm": arm,
           "atoms": atoms}
    if m < 12 and arm != "noop" and comps is not None:
        out["variants"] = phi_variants(comps)
    return out


# ---- K3: natural prediction -----------------------------------------

def k3_unit(args):
    m, cand, rep = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(_r(0, m))
    n = sim.make_initial_state(_r(1, m))
    rng = _r(5, cand_i, m, rep)
    hs = []
    record_early = []
    state40 = None
    for f in range(1, STEER + 1):
        if 21 <= f <= 40:
            d, h = traced_step(n, beta, cand, rng, record_early)
            if d is None:
                break
            hs.append(h)
            n = d
        else:
            step = sim.run_fissions(n, beta, cand, 1, rng)
            if step["n_done"] < 1:
                break
            hs.append(float(step["H"][0]))
            n = step["final"]
        if f == 40:
            state40 = n.copy()
    hs = np.array(hs)
    if len(hs) < 60 or state40 is None:
        return None
    inh = hs > sim.H_THRESH
    X9 = Ft.direct9(40, 100, hs[:40], int(state40.sum()))
    v2 = float(RI.score_states(RI._BUNDLES[cand], X9,
                               [Ft.graph_state_195(state40, beta)])[0])
    return {"matrix": m, "candidate": cand, "rep": rep,
            "phi_early": float(PC.phi_r_code(
                np.array(record_early, dtype=np.float64))),
            "v2_risk": v2,
            "hist": float(np.mean(inh[20:40])),
            "breaks_late": int((~inh[40:60]).sum())}


# ---- stats helpers ---------------------------------------------------

def mat_boot(vals, fn, n=BOOT_N, seed=BOOT_SEED):
    """vals: {matrix: payload}; fn(list_of_payloads) -> statistic."""
    mats = sorted(vals)
    point = fn([vals[m] for m in mats])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rng.choice(mats, size=len(mats), replace=True)
        bs.append(fn([vals[m] for m in pick]))
    return (float(point), float(np.nanquantile(bs, 0.025)),
            float(np.nanquantile(bs, 0.975)))


def spearman_cells(rows, xkey, ykey, centered):
    if centered:
        mx = {}
        for r in rows:
            mx.setdefault(r["matrix"], []).append(r)
        rows2 = []
        for m, rs in mx.items():
            xb = np.mean([r[xkey] for r in rs])
            yb = np.mean([r[ykey] for r in rs])
            rows2 += [{"matrix": m, xkey: r[xkey] - xb,
                       ykey: r[ykey] - yb} for r in rs]
        rows = rows2
    per = {}
    for r in rows:
        per.setdefault(r["matrix"], []).append((r[xkey], r[ykey]))

    def fn(groups):
        xs = [x for g in groups for x, _ in g]
        ys = [y for g in groups for _, y in g]
        if np.std(xs) == 0 or np.std(ys) == 0:
            return np.nan
        return spearmanr(xs, ys).correlation
    return mat_boot(per, fn)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = _ENT

    results = {}

    # ---------------- K1 ---------------------------------------------
    t0 = time.time()
    jobs = [(m, c, r, a) for c in cohort.CANDIDATES
            for m in range(N_MAT) for r in REPS for a in ARMS]
    with Pool(12) as pool:
        k1 = pool.map(k1_unit, jobs)
    print(f"K1 dose campaign in {time.time()-t0:.0f}s", flush=True)

    # ---------------- K2/K4 (Phase J replay) -------------------------
    t0 = time.time()
    jobs = [(m, c, r, a) for c in cohort.CANDIDATES
            for m in range(J.N_MAT) for r in J.REPS
            for a in ("ph_stab", "ph_destab", "noop")]
    with Pool(12) as pool:
        k2 = pool.map(k2_unit, jobs)
    print(f"K2/K4 replay in {time.time()-t0:.0f}s", flush=True)

    # replay + identity gates vs stored Phase J units
    with open(os.path.join(HERE, "results_phir_confirm",
                           "phir_confirm_units.pkl"), "rb") as f:
        jstored = pickle.load(f)
    jk = {(u["matrix"], u["candidate"], u["rep"], u["arm"]):
          u["phi_code"] for u in jstored}
    mism = 0
    for u in k2:
        if u["atoms"] is None:
            continue
        ref = jk[(u["matrix"], u["candidate"], u["rep"], u["arm"])]
        mine = u["atoms"]["phi_r"]
        if not (abs(mine - ref) < 1e-9
                or (np.isnan(ref) and np.isnan(mine))):
            mism += 1
    print(f"K2 REPLAY GATE: {'PASS' if mism == 0 else 'FAIL'} "
          f"({mism} mismatches)", flush=True)
    results["k2_replay_gate"] = bool(mism == 0)

    # ---------------- K3 ---------------------------------------------
    t0 = time.time()
    jobs = [(m, c, r) for c in cohort.CANDIDATES
            for m in range(N_MAT_NAT) for r in REPS]
    with Pool(12) as pool:
        k3 = [u for u in pool.map(k3_unit, jobs) if u]
    print(f"K3 natural cohort in {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(OUT, "phase_k_units.pkl"), "wb") as f:
        pickle.dump({"k1": k1, "k2": k2, "k3": k3}, f, protocol=4)

    for cand in cohort.CANDIDATES:
        entry = {}
        # K1
        cu = [u for u in k1 if u["candidate"] == cand]
        rung = {}
        for u in cu:
            rung.setdefault(u["arm"], []).append(u)
        entry["k1_rungs"] = {a: {
            "dose": float(np.mean([u["dose"] for u in rung[a]])),
            "inherit": float(np.nanmean([u["inherit"]
                                         for u in rung[a]])),
            "phi": float(np.nanmean([u["phi"] for u in rung[a]]))}
            for a in ARMS}
        per = {}
        for u in cu:
            per.setdefault(u["matrix"], {}).setdefault(
                u["arm"], []).append(u)

        def rho_fn(groups, key):
            rhos = []
            for g in groups:
                xs = [np.mean([u["dose"] for u in g[a]])
                      for a in ARMS if a in g]
                ys = [np.nanmean([u[key] for u in g[a]])
                      for a in ARMS if a in g]
                ok = np.isfinite(xs) & np.isfinite(ys)
                if ok.sum() > 3:
                    rhos.append(spearmanr(
                        np.array(xs)[ok], np.array(ys)[ok])
                        .correlation)
            return float(np.nanmean(rhos)) if rhos else np.nan
        for key, name in (("phi", "K1_dose_spearman_PHI"),
                          ("inherit", "K1_dose_spearman_INHERIT")):
            d, lo, hi = mat_boot(per, lambda g, k=key: rho_fn(g, k))
            entry[name] = {"rho": d, "ci": [lo, hi],
                           "excludes0": bool(lo > 0 or hi < 0)}
        # K2
        cu2 = [u for u in k2 if u["candidate"] == cand
               and u["atoms"] is not None]
        per_atom = {}
        for u in cu2:
            per_atom.setdefault(u["matrix"], {}).setdefault(
                u["arm"], []).append(u["atoms"])
        keys = [k for k in cu2[0]["atoms"]]

        def contrast_fn(groups, key):
            ds = []
            for g in groups:
                if "ph_stab" in g and "ph_destab" in g:
                    ds.append(np.nanmean([a[key] for a in g["ph_stab"]])
                              - np.nanmean([a[key]
                                            for a in g["ph_destab"]]))
            return float(np.nanmean(ds)) if ds else np.nan
        entry["k2_atoms"] = {}
        for key in keys:
            d, lo, hi = mat_boot(per_atom,
                                 lambda g, k=key: contrast_fn(g, k))
            entry["k2_atoms"][key] = {"diff": d, "ci": [lo, hi],
                                      "excludes0": bool(lo > 0
                                                        or hi < 0)}
        # K4
        cu4 = [u for u in k2 if u["candidate"] == cand
               and "variants" in u]
        entry["k4_variants"] = {}
        for var in ("no_clr", "median_agg", "drop_last"):
            per_v = {}
            for u in cu4:
                if var in u.get("variants", {}):
                    per_v.setdefault(u["matrix"], {}).setdefault(
                        u["arm"], []).append(u["variants"][var])

            def vfn(groups):
                ds = []
                for g in groups:
                    if "ph_stab" in g and "ph_destab" in g:
                        ds.append(np.nanmean(g["ph_stab"])
                                  - np.nanmean(g["ph_destab"]))
                return float(np.nanmean(ds)) if ds else np.nan
            if per_v:
                d, lo, hi = mat_boot(per_v, vfn)
                entry["k4_variants"][var] = {"diff": d,
                                             "ci": [lo, hi]}
        # K3
        cu3 = [u for u in k3 if u["candidate"] == cand]
        entry["k3"] = {}
        for pred in ("phi_early", "v2_risk", "hist"):
            for centered in (False, True):
                d, lo, hi = spearman_cells(cu3, pred, "breaks_late",
                                           centered)
                entry["k3"][f"{pred}_{'centered' if centered else 'overall'}"] = \
                    {"rho": d, "ci": [lo, hi],
                     "excludes0": bool(lo > 0 or hi < 0)}
        results[cand] = entry

        print(f"\n=== Phase K candidate {cand} ===")
        print("K1 rungs (dose, inherit, phi):")
        for a in ARMS:
            r = entry["k1_rungs"][a]
            print(f"  {a:9s} {r['dose']:+.3f} {r['inherit']:.3f} "
                  f"{r['phi']:.4f}")
        for name in ("K1_dose_spearman_PHI", "K1_dose_spearman_INHERIT"):
            v = entry[name]
            print(f"{name}: {v['rho']:+.3f} "
                  f"CI [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                  + ("  *" if v["excludes0"] else ""))
        sig = {k: v for k, v in entry["k2_atoms"].items()
               if v["excludes0"] and k not in ("phi_r",)}
        print("K2 atoms with stab-destab CIs excluding 0:")
        for k, v in sorted(sig.items(), key=lambda kv: -abs(
                kv[1]["diff"])):
            print(f"  {k:14s} {v['diff']:+.4f} "
                  f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
        for k in ("phi_r", "synergy", "causation", "emergence"):
            v = entry["k2_atoms"][k]
            print(f"K2 {k}: {v['diff']:+.4f} "
                  f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
                  + ("  *" if v["excludes0"] else ""))
        for var, v in entry["k4_variants"].items():
            print(f"K4 {var}: {v['diff']:+.4f} "
                  f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
        for k, v in entry["k3"].items():
            print(f"K3 {k}: {v['rho']:+.3f} "
                  f"[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                  + ("  *" if v["excludes0"] else ""))

    with open(os.path.join(OUT, "phase_k_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "phase_k_results.json"))


if __name__ == "__main__":
    main()
