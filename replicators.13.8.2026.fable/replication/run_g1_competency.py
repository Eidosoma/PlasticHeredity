"""Phase G1: competency-attractor test (preregistered in PHASE_G.md).

Does BEHAVIOR return after perturbation even though composition does
not? Behavioral fingerprint K(s) (8 components, branch-estimated,
z-frozen from dev matrices); five state classes; two-target race:
distance of K(t) back to the state's own K(0) versus to the
matrix-typical fingerprint. Registered prediction: matrix-typical
wins (distributional relaxation, no state-specific competency memory).
Singh-Jain behavioral positive control gates the conclusion.
"""

import json
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

import sim
import features as Ft
import cohort
import run_intervention as RI
import run_steering as RS
import run_f7_attractor_controller as F7
import sj_model as SJ
from run_f3_f4 import kswap_perturb
from run_d1_outcomes import extended_branch_outcomes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_g")
TAG = "steering-2026-08-13"
DEV_TAG = "25x-2026-08-13"
N_MAT = 24
N_FP = 16                      # branches per fingerprint
TIMES = [1, 2, 5, 10, 24]
CLASSES = ["natural", "model_down", "model_up", "comp_aligned",
           "ordinary"]
ATYPICAL = ["model_down", "model_up", "comp_aligned"]
EPS = 1e-12

FP_NAMES = ["qb4", "qb8", "qb12", "qjoint", "qpersist5",
            "inherited", "updates", "dentropy"]

_Z = None          # frozen (mean, sd) per component


def entropy_of(n):
    x = n[n > 0] / n.sum()
    return float(-np.sum(x * np.log(x + EPS)))


def fingerprint(s, beta, cand, key, n_br=N_FP):
    """K(s) from n_br independent 12-fission branches (domain 20)."""
    e0 = entropy_of(s)
    comp = {k: [] for k in FP_NAMES}
    for b in range(n_br):
        rb = cohort._rng(RI._ENT, 20, *key, b)
        br = sim.run_fissions(s, beta, cand, 12, rb)
        inh = br["inherited"]
        eo = extended_branch_outcomes(inh, br["died"])
        comp["qb4"].append(float((~inh[:4]).any()) if len(inh) >= 4
                           else np.nan)
        comp["qb8"].append(float((~inh[:8]).any()) if len(inh) >= 8
                           else np.nan)
        comp["qb12"].append(eo["break"])
        comp["qjoint"].append(float(Ft.joint_break_run3(inh)))
        comp["qpersist5"].append(eo["persist5"])
        comp["inherited"].append(eo["inherited_count"])
        comp["updates"].append(float(np.mean(br["updates"]))
                               if br["n_done"] else np.nan)
        comp["dentropy"].append(
            entropy_of(br["daughters"][-1]) - e0 if br["n_done"]
            else np.nan)
    return np.array([np.nanmean(comp[k]) for k in FP_NAMES])


def zdist(a, b):
    """NaN-robust z-distance: use mutually finite components, rescaled
    to the full 8-component length for comparability."""
    z = (a - b) / _Z[1]
    m = np.isfinite(z)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.sum(z[m] ** 2) * (len(z) / m.sum())))


def dev_standardization():
    """Frozen z-scaling from a dev fingerprint cohort."""
    ent = cohort.domain_entropy("dev", DEV_TAG)
    rows = []
    for cand in cohort.CANDIDATES:
        cand_i = cohort.CANDIDATES.index(cand)
        for m in range(12):
            beta, n0 = cohort.matrix_and_init(ent, m)
            rng = cohort._rng(ent, 2, cand_i, m)
            tr = sim.run_fissions(n0, beta, cand, 80, rng)
            for lm in (20, 50, 80):
                if lm > tr["n_done"]:
                    continue
                old = RI._ENT
                RI._ENT = ent
                rows.append(fingerprint(tr["daughters"][lm - 1], beta,
                                        cand, (900, cand_i, m, lm),
                                        n_br=8))
                RI._ENT = old
    R = np.array(rows)
    return (np.nanmean(R, axis=0),
            np.maximum(np.nanstd(R, axis=0), 1e-6))


def make_states(m, cand):
    """Deterministically regenerate the five state classes."""
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    states = {}
    rn = cohort._rng(RI._ENT, 7, cand_i, m, 0)
    nat = sim.run_fissions(n0, beta, cand, 60, rn)
    states["natural"] = nat["daughters"][59]
    states["ordinary"] = nat["daughters"][29]
    for cls, ctrl in [("model_down", "model_down"),
                      ("model_up", "model_up")]:
        holder = {}
        RS.steer_lineage(n0, beta, cand, cand_i, m, 0, ctrl,
                         log=lambda f, n, s, H, u: holder.update(n=n))
        states[cls] = holder["n"]
    # comp_aligned: regenerate F7's comp_only lineage (rep 0)
    cfg = "frozen02" if cand == "02" else "frozen03"
    atl = F7._ATLASES[(cfg, m)]
    rng = cohort._rng(RI._ENT, 16, cand_i, m, 0)
    n = n0.copy()
    hs = []
    for f in range(1, 61):
        step = sim.run_fissions(n, beta, cand, 1, rng)
        if step["n_done"] < 1:
            break
        hs.append(float(step["H"][0]))
        n = step["final"]
        if f == 60:
            break
        X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
        n = RI.apply_swap(n, F7.objective_swap("comp_only", n, beta,
                                               cand, atl, X9))
    states["comp_aligned"] = n
    return beta, states


def g1_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, states = make_states(m, cand)

    # matrix-typical fingerprint (independent probe lineage)
    rngp = cohort._rng(RI._ENT, 20, 800, cand_i, m)
    probe = sim.run_fissions(sim.make_initial_state(rngp), beta, cand,
                             60, rngp)
    kmats = []
    for i, lm in enumerate((20, 35, 50)):
        if lm <= probe["n_done"]:
            kmats.append(fingerprint(probe["daughters"][lm - 1], beta,
                                     cand, (801, cand_i, m, i), n_br=8))
    K_matrix = np.mean(kmats, axis=0)

    out = {}
    for cls_i, cls in enumerate(CLASSES):
        s = states[cls]
        K0 = fingerprint(s, beta, cand, (0, cand_i, m, cls_i, 0))
        traj = {}
        for ai, arm in enumerate(("none", "k8")):
            if arm == "k8":
                pr = cohort._rng(RI._ENT, 20, 700, cand_i, m, ai)
                s0 = kswap_perturb(s, 8, pr)
            else:
                s0 = s
            dists0, distsM = {t: [] for t in TIMES}, {t: [] for t in TIMES}
            for c in range(3):
                rc = cohort._rng(RI._ENT, 20, 1, cand_i, m, cls_i, ai, c)
                carrier = sim.run_fissions(s0, beta, cand, 24, rc)
                for ti, t in enumerate(TIMES):
                    if t > carrier["n_done"]:
                        continue
                    Kt = fingerprint(carrier["daughters"][t - 1], beta,
                                     cand, (2, cand_i, m, cls_i, ai, c,
                                            ti))
                    dists0[t].append(zdist(Kt, K0))
                    distsM[t].append(zdist(Kt, K_matrix))
            traj[arm] = {
                "d0": {t: float(np.mean(v)) for t, v in dists0.items()
                       if v},
                "dM": {t: float(np.mean(v)) for t, v in distsM.items()
                       if v},
            }
        out[cls] = {"K0_dist_to_matrix": zdist(K0, K_matrix),
                    "traj": traj}
    return {"matrix": m, "candidate": cand, "out": out}


def sj_behavioral_control():
    """S-J positive control: within-basin behavioral return."""
    def fp_sj(X, seedbase, n_br=12, hor=5):
        vals = []
        for b in range(n_br):
            o = SJ.run_lineage(np.array(X, dtype=np.int64), hor,
                               np.random.default_rng(seedbase + b))
            if len(o["modes"]) == hor:
                vals.append([o["modes"][-1], o["pre"][-1][2],
                             float(np.mean(o["taus"]))])
        return np.nanmean(np.array(vals, dtype=float), axis=0)

    hits = 0
    total = 0
    for li in range(12):
        rng = np.random.default_rng(10_000 + li * 2 + 1)
        o = SJ.run_lineage(SJ.ACTIVE_INIT, 21, rng)
        # first ACTIVE division at index >= 5 (short residence makes a
        # fixed-division filter starve the control of states)
        idx = next((k for k in range(5, len(o["modes"]))
                    if o["modes"][k] == 1), None)
        if idx is None:
            continue
        X = o["pre"][idx]
        K0 = fp_sj(X, 3_000_000 + li * 500)
        Xp = X.copy()
        Xp[2] = max(Xp[2] - 2, 0)
        Kt = fp_sj(Xp, 3_100_000 + li * 500)
        Kin = fp_sj(SJ.INACTIVE_INIT, 3_200_000 + li * 500)
        total += 1
        if np.linalg.norm(Kt - K0) < np.linalg.norm(Kt - Kin):
            hits += 1
    return hits / max(total, 1), total


def main():
    global _Z
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    with open(os.path.join(HERE, "results_f", "atlases.pkl"), "rb") as f:
        F7._ATLASES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)

    t0 = time.time()
    _Z = dev_standardization()
    print(f"dev z-standardization in {time.time()-t0:.0f}s")

    frac, n = sj_behavioral_control()
    print(f"S-J behavioral positive control: within-basin behavioral "
          f"return {frac:.2f} (n={n}) -> "
          f"{'PASS' if frac >= 0.8 else 'FAIL'}")

    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MAT)]
    with Pool(12) as pool:
        units = pool.map(g1_unit, jobs)
    print(f"G1 campaign in {time.time()-t0:.0f}s")
    with open(os.path.join(OUT, "g1_units.pkl"), "wb") as f:
        pickle.dump({"units": units, "Z": _Z}, f, protocol=4)

    results = {"sj_control": {"frac": frac, "n": n,
                              "pass": bool(frac >= 0.8)},
               "z_mean": _Z[0].tolist(), "z_sd": _Z[1].tolist()}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        mats = np.array([u["matrix"] for u in cu])
        entry = {}
        for cls in CLASSES:
            e = {"K0_dist_to_matrix": float(np.mean(
                [u["out"][cls]["K0_dist_to_matrix"] for u in cu]))}
            for arm in ("none", "k8"):
                d0 = {t: float(np.mean(
                    [u["out"][cls]["traj"][arm]["d0"].get(t, np.nan)
                     for u in cu])) for t in TIMES}
                dM = {t: float(np.mean(
                    [u["out"][cls]["traj"][arm]["dM"].get(t, np.nan)
                     for u in cu])) for t in TIMES}
                e[arm] = {"d_to_K0": d0, "d_to_matrix": dM}
            entry[cls] = e
        # two-target race for atypical classes at t=10, k8 arm
        race = {}
        for cls in ATYPICAL:
            d0_10 = np.array([u["out"][cls]["traj"]["k8"]["d0"].get(10,
                              np.nan) for u in cu])
            dM_10 = np.array([u["out"][cls]["traj"]["k8"]["dM"].get(10,
                              np.nan) for u in cu])
            peak = np.array([max(u["out"][cls]["traj"]["k8"]["d0"]
                                 .values()) for u in cu])
            diff = dM_10 - d0_10          # >0 means closer to K0
            ok = np.isfinite(diff)
            rng = np.random.default_rng(2024)
            ci = (RI.boot_lower(diff[ok], mats[ok], rng, n=1024)
                  if ok.any() else (np.nan, np.nan))
            race[cls] = {
                "d_to_K0_at10": float(np.nanmean(d0_10)),
                "d_to_matrix_at10": float(np.nanmean(dM_10)),
                "halved_vs_peak": bool(np.nanmean(d0_10)
                                       <= 0.5 * np.nanmean(peak)),
                "closer_to_K0_diff": {"mean": float(np.nanmean(diff)),
                                      "ci": ci},
                "competency_return": bool(
                    np.nanmean(d0_10) <= 0.5 * np.nanmean(peak)
                    and np.nanmean(diff) > 0 and ci[0] > 0),
            }
        entry["race"] = race
        results[cand] = entry

        print(f"\n=== G1 candidate {cand} ===")
        for cls in CLASSES:
            e = entry[cls]
            print(f"{cls:12s} K0-vs-matrix dist "
                  f"{e['K0_dist_to_matrix']:.2f} | k8 d_to_K0 "
                  + "/".join(f"{e['k8']['d_to_K0'][t]:.2f}"
                             for t in TIMES)
                  + " | d_to_matrix "
                  + "/".join(f"{e['k8']['d_to_matrix'][t]:.2f}"
                             for t in TIMES))
        for cls, r in race.items():
            print(f"race {cls:12s}: @10 d(K0) {r['d_to_K0_at10']:.2f} vs "
                  f"d(matrix) {r['d_to_matrix_at10']:.2f} | "
                  f"closer-to-K0 diff {r['closer_to_K0_diff']['mean']:+.2f} "
                  f"CI [{r['closer_to_K0_diff']['ci'][0]:+.2f},"
                  f"{r['closer_to_K0_diff']['ci'][1]:+.2f}] | "
                  f"COMPETENCY RETURN: {r['competency_return']}")

    with open(os.path.join(OUT, "g1_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "g1_results.json"))


if __name__ == "__main__":
    main()
