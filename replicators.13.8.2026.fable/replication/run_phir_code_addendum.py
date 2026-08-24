"""Phase I ADDENDUM: exact deterministic replay of the frozen bridge
campaign, re-measured with the code-faithful Phi_R (phir_code.py).
Registered in PHIR_BRIDGE.md ADDENDUM. The lineage loop MUST mirror
run_phir_bridge.unit exactly; the replay gate (equality of heredity
outcomes and text-formula Phi-r against the stored units) enforces
this."""

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
import run_phir_bridge as B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_phir_bridge")


def unit(args):
    """Mirror of run_phir_bridge.unit with both Phi instruments."""
    m, cand, rep, arm = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(B._r(0, m))
    n = sim.make_initial_state(B._r(1, m))
    rng = B._r(2, cand_i, m, rep)
    hs, record = [], []
    for f in range(1, B.STEER + 1):
        if f >= B.PHI_FROM:
            d, h = B.traced_step(n, beta, cand, rng, record)
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
        if f == B.STEER:
            break
        panel = B.draw_panel(n, B._r(3, cand_i, m, rep, f))
        if arm == "noop":
            continue
        if arm == "random":
            pick = panel[int(B._r(4, cand_i, m, rep, f)
                             .integers(B.PANEL))]
        elif arm in ("ph_stab", "ph_destab"):
            sc = B.v2_scores(n, beta, cand, hs, f, panel)
            pick = panel[int(np.argmin(sc) if arm == "ph_stab"
                             else np.argmax(sc))]
        else:
            sc = B.surr_scores(n, beta, panel)
            pick = panel[int(np.argmax(sc) if arm == "phir_max"
                             else np.argmin(sc))]
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
            "phi_text": float(phir.phi_r_series(comps))
            if comps is not None else np.nan,
            "phi_code": float(phir_code.phi_r_code(comps))
            if comps is not None else np.nan}


def main():
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = B._ENT

    t0 = time.time()
    jobs = [(m, c, r, a) for c in cohort.CANDIDATES
            for m in range(B.N_MAT) for r in B.REPS for a in B.ARMS]
    with Pool(12) as pool:
        units = pool.map(unit, jobs)
    print(f"addendum replay in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "phir_code_units.pkl"), "wb") as f:
        pickle.dump(units, f, protocol=4)

    with open(os.path.join(OUT, "phir_bridge_units.pkl"), "rb") as f:
        stored = pickle.load(f)
    skey = {(u["matrix"], u["candidate"], u["rep"], u["arm"]): u
            for u in stored}
    mismatch = 0
    for u in units:
        s = skey[(u["matrix"], u["candidate"], u["rep"], u["arm"])]
        same = (u["inherit"] == s["inherit"]
                and u["breaks"] == s["breaks"]
                and u["longest_run"] == s["longest_run"]
                and (u["phi_text"] == s["phi"]
                     or (np.isnan(u["phi_text"])
                         and np.isnan(s["phi"]))))
        mismatch += 0 if same else 1
    replay_ok = mismatch == 0
    print(f"REPLAY GATE: {'PASS' if replay_ok else 'FAIL'} "
          f"({mismatch} mismatching units of {len(units)})")

    results = {"replay_gate": bool(replay_ok)}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        entry = {"arms": {}, "tests": {}}
        for arm in B.ARMS:
            au = [u for u in cu if u["arm"] == arm]
            entry["arms"][arm] = {
                "phi_code": float(np.nanmean([u["phi_code"]
                                              for u in au])),
                "phi_text": float(np.nanmean([u["phi_text"]
                                              for u in au]))}
        for name, a, b in (
                ("T1code_phstab_minus_phdestab", "ph_stab",
                 "ph_destab"),
                ("MANIPcode_phirmax_minus_phirmin", "phir_max",
                 "phir_min"),
                ("T4code_random_minus_noop", "random", "noop")):
            d, lo, hi = B.boot_pairs(cu, a, b, "phi_code")
            entry["tests"][name] = {"diff": d, "ci": [lo, hi],
                                    "excludes0": bool(lo > 0
                                                      or hi < 0)}
        tv = [u["phi_text"] for u in cu]
        cv = [u["phi_code"] for u in cu]
        ok = np.isfinite(tv) & np.isfinite(cv)
        entry["text_code_pearson"] = float(np.corrcoef(
            np.array(tv)[ok], np.array(cv)[ok])[0, 1])
        results[cand] = entry

        print(f"\n=== ADDENDUM candidate {cand} ===")
        print(f"{'arm':10s} {'phi_code':>9s} {'phi_text':>9s}")
        for arm in B.ARMS:
            a = entry["arms"][arm]
            print(f"{arm:10s} {a['phi_code']:9.4f} "
                  f"{a['phi_text']:9.4f}")
        for k, v in entry["tests"].items():
            print(f"{k}: {v['diff']:+.4f} "
                  f"CI [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
                  + ("  *" if v["excludes0"] else ""))
        print(f"text-vs-code Pearson across lineages: "
              f"{entry['text_code_pearson']:+.3f}")

    with open(os.path.join(OUT, "phir_code_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "phir_code_results.json"))


if __name__ == "__main__":
    main()
