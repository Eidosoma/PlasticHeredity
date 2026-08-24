"""Phase F5a: Singh-Jain bistable protocell as the POSITIVE CONTROL
for the basin assay (assay-validity gate).

The S-J model has two analytically known basins (ACS-inactive vs
ACS-active) at kappa=2400. The assay framework used on GARD
(perturb -> free branches -> classify hold/switch; plus a radial-drift
estimator) must detect them here, or GARD negatives are suspended.

Registered gates (assay-validity):
  G1  free-lineage mode inheritance >= 0.9 (daughters keep the
      mother's mode) and both modes observed with at least one
      spontaneous switch across the lineage ensemble;
  G2  basin hold: P(same mode after 16 divisions | no perturbation)
      >= 0.8 for states in both modes;
  G3  dose response: for active-mode states, hold probability under
      full catalyst knockout (X4 -> 0) is lower than under a -2
      knockout by >= 0.2;
  G4  restoring drift: for active-mode states, the one-division drift
      of |log10(X4+1) - log10(X4_active_mean+1)| is negative
      (restoring) under small perturbations.

Also documented (not gated): the GARD-style raw-cosine classifier on
concentration vectors — expected to be INSENSITIVE to these basins
(x4 is a numerically tiny component), an important lesson about
metric choice, not an assay failure.
"""

import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np

import sj_model as SJ

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_f")
N_LIN = 24
N_DIV_FREE = 60
N_STATES = 16
ARMS = ["none", "x4_minus2", "x4_minus10", "x4_zero", "x4_plus10"]
N_BR = 16
HOR = 16


def perturb_sj(X, arm):
    X = X.copy()
    if arm == "x4_minus2":
        X[2] = max(X[2] - 2, 0)
    elif arm == "x4_minus10":
        X[2] = max(X[2] - 10, 0)
    elif arm == "x4_zero":
        X[2] = 0
    elif arm == "x4_plus10":
        X[2] += 10
    return X


def free_lineage_unit(args):
    li, start = args
    init = SJ.ACTIVE_INIT if start == "active" else SJ.INACTIVE_INIT
    rng = np.random.default_rng(10_000 + li * 2
                                + (1 if start == "active" else 0))
    out = SJ.run_lineage(init, N_DIV_FREE, rng)
    modes = out["modes"]
    inherit = float(np.mean(modes[1:] == modes[:-1])) if len(modes) > 1 \
        else np.nan
    switches = int(np.sum(modes[1:] != modes[:-1]))
    return {"start": start, "modes": modes.tolist(),
            "inherit": inherit, "switches": switches,
            "taus": out["taus"].tolist(),
            "pre20": out["pre"][19].tolist() if len(out["pre"]) > 19
            else None}


def basin_unit(args):
    mode, si, state = args
    X = np.array(state, dtype=np.int64)
    res = {}
    for ai, arm in enumerate(ARMS):
        Xp = perturb_sj(X, arm)
        holds = []
        for b in range(N_BR):
            rng = np.random.default_rng(
                500_000 + si * 1000 + ai * 100 + b
                + (50_000 if mode == "active" else 0))
            out = SJ.run_lineage(Xp, HOR, rng)
            if len(out["modes"]) == HOR:
                final = out["modes"][-1]
                holds.append(int(final == (1 if mode == "active" else 0)))
        res[arm] = float(np.mean(holds)) if holds else np.nan
    # G4: one-division log-x4 drift under small perturbation (-2)
    drifts = []
    x4_ref = X[2]
    for b in range(N_BR):
        rng = np.random.default_rng(900_000 + si * 100 + b)
        out = SJ.run_lineage(perturb_sj(X, "x4_minus2"), 1, rng)
        if len(out["pre"]):
            d0 = abs(np.log10(perturb_sj(X, "x4_minus2")[2] + 1)
                     - np.log10(x4_ref + 1))
            d1 = abs(np.log10(out["pre"][0][2] / 2 + 1)
                     - np.log10(x4_ref + 1))
            drifts.append(d1 - d0)
    res["log_drift"] = float(np.mean(drifts)) if drifts else np.nan
    # raw-cosine (GARD-style) sensitivity check
    a = SJ.concentrations(np.array(SJ.ACTIVE_INIT))
    i = SJ.concentrations(np.array(SJ.INACTIVE_INIT))
    res["cosine_between_modes"] = float(
        np.dot(a, i) / (np.linalg.norm(a) * np.linalg.norm(i)))
    return {"mode": mode, "res": res}


def main():
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()
    jobs = [(li, s) for s in ("inactive", "active") for li in range(N_LIN)]
    with Pool(12) as pool:
        lins = pool.map(free_lineage_unit, jobs)
    print(f"free lineages in {time.time()-t0:.0f}s")

    inherit = float(np.nanmean([l["inherit"] for l in lins]))
    total_switches = int(np.sum([l["switches"] for l in lins]))
    seen_active = any(1 in l["modes"] for l in lins)
    seen_inactive = any(0 in l["modes"] for l in lins)
    tau_by_mode = {m: [] for m in (0, 1)}
    for l in lins:
        for md, tau in zip(l["modes"], l["taus"]):
            tau_by_mode[md].append(tau)
    g1 = bool(inherit >= 0.9 and total_switches >= 1
              and seen_active and seen_inactive)
    print(f"G1 mode inheritance {inherit:.3f}, switches "
          f"{total_switches}, both modes seen "
          f"{seen_active and seen_inactive} -> {'PASS' if g1 else 'FAIL'}")
    print(f"interdivision tau: inactive {np.mean(tau_by_mode[0]):.3f} "
          f"vs active {np.mean(tau_by_mode[1]):.3f}")

    # basin states: division-20 pre-states matching the intended mode
    states = {"inactive": [], "active": []}
    for l in lins:
        if l["pre20"] is None:
            continue
        md = "active" if l["modes"][19] == 1 else "inactive"
        if len(states[md]) < N_STATES:
            states[md].append(l["pre20"])
    print(f"basin states: inactive {len(states['inactive'])}, "
          f"active {len(states['active'])}")

    t0 = time.time()
    jobs = [(m, si, s) for m in ("inactive", "active")
            for si, s in enumerate(states[m])]
    with Pool(12) as pool:
        basins = pool.map(basin_unit, jobs)
    print(f"basin assay in {time.time()-t0:.0f}s")

    results = {"G1": {"inherit": inherit, "switches": total_switches,
                      "pass": g1},
               "tau": {"inactive": float(np.mean(tau_by_mode[0])),
                       "active": float(np.mean(tau_by_mode[1]))}}
    holds = {}
    for m in ("inactive", "active"):
        bu = [b["res"] for b in basins if b["mode"] == m]
        holds[m] = {arm: float(np.nanmean([r[arm] for r in bu]))
                    for arm in ARMS}
        results[f"hold_{m}"] = holds[m]
        print(f"hold {m:8s}: " + " | ".join(
            f"{arm} {holds[m][arm]:.2f}" for arm in ARMS))
    g2 = bool(holds["inactive"]["none"] >= 0.8
              and holds["active"]["none"] >= 0.8)
    g3 = bool(holds["active"]["x4_minus2"] - holds["active"]["x4_zero"]
              >= 0.2)
    act_drift = float(np.nanmean([b["res"]["log_drift"] for b in basins
                                  if b["mode"] == "active"]))
    g4 = bool(act_drift < 0)
    cosb = basins[0]["res"]["cosine_between_modes"]
    results.update({"G2_pass": g2, "G3_pass": g3,
                    "G4_active_log_drift": act_drift, "G4_pass": g4,
                    "cosine_between_mode_concentrations": cosb})
    print(f"G2 hold-under-none pass: {g2} | G3 dose-response pass: {g3} "
          f"| G4 restoring log-drift {act_drift:+.4f} pass: {g4}")
    print(f"raw-cosine between mode concentration vectors: {cosb:.4f} "
          f"(GARD-style metric {'CANNOT' if cosb > 0.9 else 'can'} "
          f"separate these basins)")
    results["assay_validity_pass"] = bool(g1 and g2 and g3 and g4)
    print(f"\nASSAY VALIDITY: "
          f"{'PASS' if results['assay_validity_pass'] else 'FAIL'}")

    with open(os.path.join(OUT, "f5_sj_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("written:", os.path.join(OUT, "f5_sj_results.json"))


if __name__ == "__main__":
    main()
