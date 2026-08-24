"""Phase J: prospective Φ_R signature-versus-controller test at 2x
scale (preregistered in PHIR_CONFIRM.md; domain 28; sealed before
the campaign). Probe-rollout Φ_R controller — no surrogate."""

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
import phir
import phir_code
import run_intervention as RI
from run_phir_bridge import traced_step, v2_scores

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_phir_confirm")
TAG = "phir-confirm-2026-08-17"
DOMAIN = 28
N_MAT = 48
REPS = [0, 1]
STEER = 60
PHI_FROM = 41
PANEL = 12
PROBE_FISSIONS = 2
ARMS = ["ph_stab", "ph_destab", "phiR_max", "phiR_min", "random",
        "noop"]
BOOT_N = 4096
BOOT_SEED = 17

_ENT = cohort.domain_entropy("confirmation", TAG)


def _r(*key):
    return cohort._rng(_ENT, DOMAIN, *key)


def draw_panel(n, rng):
    present = np.where(n > 0)[0]
    panel = []
    for _ in range(PANEL):
        i = int(present[rng.integers(len(present))])
        j = int(rng.integers(sim.NG - 1))
        panel.append((i, j + 1 if j >= i else j))
    return panel


def probe_phi(n0, beta, cand, cand_i, m, rep, f):
    """Probed code-Phi_R of state n0: 2 fresh fissions on the CRN
    probe stream (identical for every edit at this decision),
    Phi_R on the probe's update series. NaN on death/short series."""
    rng = _r(5, cand_i, m, rep, f)
    record = []
    n = n0
    for _ in range(PROBE_FISSIONS):
        d, h = traced_step(n, beta, cand, rng, record)
        if d is None:
            return np.nan
        n = d
    if not record:
        return np.nan
    return phir_code.phi_r_code(np.array(record, dtype=np.float64))


def unit(args):
    m, cand, rep, arm = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(_r(0, m))
    n = sim.make_initial_state(_r(1, m))
    rng = _r(2, cand_i, m, rep)          # CRN, arm-independent
    hs, record = [], []
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
        panel = draw_panel(n, _r(3, cand_i, m, rep, f))
        if arm == "noop":
            continue
        if arm == "random":
            pick = panel[int(_r(4, cand_i, m, rep, f).integers(PANEL))]
        elif arm in ("ph_stab", "ph_destab"):
            sc = v2_scores(n, beta, cand, hs, f, panel)
            pick = panel[int(np.argmin(sc) if arm == "ph_stab"
                             else np.argmax(sc))]
        else:
            sc = np.array([probe_phi(RI.apply_swap(n, sw), beta, cand,
                                     cand_i, m, rep, f)
                           for sw in panel])
            if np.isfinite(sc).any():
                masked = np.where(np.isfinite(sc), sc,
                                  -np.inf if arm == "phiR_max"
                                  else np.inf)
                pick = panel[int(np.argmax(masked)
                                 if arm == "phiR_max"
                                 else np.argmin(masked))]
            else:
                pick = None
        if pick is not None:
            n = RI.apply_swap(n, pick)
    hs = np.array(hs)
    inh = hs > sim.H_THRESH
    runs, cur = [0], 0
    for v in inh:
        cur = cur + 1 if v else 0
        runs.append(cur)
    comps = np.array(record, dtype=np.float64) if record else None
    return {"matrix": m, "candidate": cand, "rep": rep, "arm": arm,
            "inherit": float(inh.mean()) if len(inh) else np.nan,
            "breaks": int((~inh).sum()),
            "longest_run": int(max(runs)),
            "phi_code": float(phir_code.phi_r_code(comps))
            if comps is not None else np.nan,
            "phi_text": float(phir.phi_r_series(comps))
            if comps is not None else np.nan}


def boot_pairs(units, arm_a, arm_b, key):
    per = {}
    for u in units:
        per.setdefault((u["matrix"], u["rep"]), {})[u["arm"]] = u
    diffs = {}
    for (m, rep), d in per.items():
        if arm_a in d and arm_b in d:
            a, b = d[arm_a][key], d[arm_b][key]
            if np.isfinite(a) and np.isfinite(b):
                diffs.setdefault(m, []).append(a - b)
    means = {m: np.mean(v) for m, v in diffs.items()}
    mats = list(means)
    rng = np.random.default_rng(BOOT_SEED)
    bs = [np.mean([means[mm] for mm in
                   rng.choice(mats, size=len(mats), replace=True)])
          for _ in range(BOOT_N)]
    return (float(np.mean(list(means.values()))),
            float(np.quantile(bs, 0.025)),
            float(np.quantile(bs, 0.975)))


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = _ENT

    t0 = time.time()
    jobs = [(m, c, r, a) for c in cohort.CANDIDATES
            for m in range(N_MAT) for r in REPS for a in ARMS]
    with Pool(12) as pool:
        units = pool.map(unit, jobs)
    print(f"Phase J campaign in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "phir_confirm_units.pkl"), "wb") as f:
        pickle.dump(units, f, protocol=4)

    results = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        entry = {"arms": {}, "tests": {}}
        for arm in ARMS:
            au = [u for u in cu if u["arm"] == arm]
            entry["arms"][arm] = {
                k: float(np.nanmean([u[k] for u in au]))
                for k in ("inherit", "breaks", "longest_run",
                          "phi_code", "phi_text")}
        for name, a, b, key in (
                ("C1_phstab_minus_phdestab_PHICODE", "ph_stab",
                 "ph_destab", "phi_code"),
                ("C2a_phiRmax_minus_phiRmin_PHICODE", "phiR_max",
                 "phiR_min", "phi_code"),
                ("C2b_phiRmax_minus_phiRmin_INHERIT", "phiR_max",
                 "phiR_min", "inherit"),
                ("C2b_phiRmax_minus_phiRmin_BREAKS", "phiR_max",
                 "phiR_min", "breaks"),
                ("C2b_phiRmax_minus_phiRmin_LONGRUN", "phiR_max",
                 "phiR_min", "longest_run"),
                ("C3_phstab_minus_phdestab_INHERIT", "ph_stab",
                 "ph_destab", "inherit"),
                ("C4_random_minus_noop_INHERIT", "random", "noop",
                 "inherit"),
                ("C4_random_minus_noop_PHICODE", "random", "noop",
                 "phi_code")):
            d, lo, hi = boot_pairs(cu, a, b, key)
            entry["tests"][name] = {"diff": d, "ci": [lo, hi],
                                    "excludes0": bool(lo > 0
                                                      or hi < 0)}
        results[cand] = entry

        print(f"\n=== Phase J candidate {cand} ===")
        print(f"{'arm':9s} {'inherit':>8s} {'breaks':>7s} "
              f"{'longrun':>8s} {'phiCODE':>8s} {'phiTEXT':>8s}")
        for arm in ARMS:
            a = entry["arms"][arm]
            print(f"{arm:9s} {a['inherit']:8.3f} {a['breaks']:7.2f} "
                  f"{a['longest_run']:8.1f} {a['phi_code']:8.4f} "
                  f"{a['phi_text']:8.4f}")
        for k, v in entry["tests"].items():
            print(f"{k}: {v['diff']:+.4f} "
                  f"CI [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
                  + ("  *" if v["excludes0"] else ""))

    with open(os.path.join(OUT, "phir_confirm_results.json"),
              "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:",
          os.path.join(OUT, "phir_confirm_results.json"))


if __name__ == "__main__":
    main()
