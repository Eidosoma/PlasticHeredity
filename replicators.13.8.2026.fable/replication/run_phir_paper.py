"""Phase L: paper-faithful Phi-r on byte-exact replays of Phase J
arms (PHIR_PAPER.md; sealed). The lineage loop mirrors
run_phir_confirm.unit with recording extended to fissions 21-60;
the replay gate (code-Phi_R on the 41-60 sub-record vs stored
Phase J values) enforces exactness."""

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
import phir_code
import phir_paper
import run_intervention as RI
from run_phir_bridge import traced_step, v2_scores
import run_phir_confirm as J

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_phir_paper")
ARMS = ["ph_stab", "ph_destab", "random", "noop"]
REC_FROM = 21
BOOT_N = 4096
BOOT_SEED = 23


def unit(args):
    m, cand, rep, arm = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(J._r(0, m))
    n = sim.make_initial_state(J._r(1, m))
    rng = J._r(2, cand_i, m, rep)
    hs, record, marks = [], [], {}
    for f in range(1, J.STEER + 1):
        if f >= REC_FROM:
            marks.setdefault(f, len(record))
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
        if arm == "random":
            pick = panel[int(J._r(4, cand_i, m, rep, f)
                             .integers(J.PANEL))]
        else:
            sc = v2_scores(n, beta, cand, hs, f, panel)
            pick = panel[int(np.argmin(sc) if arm == "ph_stab"
                             else np.argmax(sc))]
        n = RI.apply_swap(n, pick)
    comps = np.array(record, dtype=np.float64) if record else None
    i41 = marks.get(J.PHI_FROM)
    out = {"matrix": m, "candidate": cand, "rep": rep, "arm": arm,
           "paper_full": np.nan, "paper_late": np.nan,
           "code_late": np.nan}
    if comps is not None:
        out["paper_full"] = float(phir_paper.phi_r_paper(comps))
        if i41 is not None:
            late = comps[i41:]
            out["paper_late"] = float(phir_paper.phi_r_paper(late))
            out["code_late"] = float(phir_code.phi_r_code(late))
    return out


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
    RI._ENT = J._ENT

    t0 = time.time()
    jobs = [(m, c, r, a) for c in cohort.CANDIDATES
            for m in range(J.N_MAT) for r in J.REPS for a in ARMS]
    with Pool(12) as pool:
        units = pool.map(unit, jobs)
    print(f"Phase L replay in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "phir_paper_units.pkl"), "wb") as f:
        pickle.dump(units, f, protocol=4)

    with open(os.path.join(HERE, "results_phir_confirm",
                           "phir_confirm_units.pkl"), "rb") as f:
        stored = pickle.load(f)
    sk = {(u["matrix"], u["candidate"], u["rep"], u["arm"]):
          u["phi_code"] for u in stored}
    mism = 0
    for u in units:
        ref = sk.get((u["matrix"], u["candidate"], u["rep"],
                      u["arm"]))
        mine = u["code_late"]
        if ref is None:
            continue
        if not (abs(mine - ref) < 1e-9
                or (np.isnan(ref) and np.isnan(mine))):
            mism += 1
    print(f"REPLAY GATE: {'PASS' if mism == 0 else 'FAIL'} "
          f"({mism} mismatches of {len(units)})", flush=True)

    results = {"replay_gate": bool(mism == 0)}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        entry = {"arms": {}, "tests": {}}
        for arm in ARMS:
            au = [u for u in cu if u["arm"] == arm]
            entry["arms"][arm] = {
                k: float(np.nanmean([u[k] for u in au]))
                for k in ("paper_full", "paper_late")}
            entry["arms"][arm]["n_valid_full"] = int(
                np.isfinite([u["paper_full"] for u in au]).sum())
        for name, a, b, key in (
                ("L1_phstab_minus_phdestab_FULL", "ph_stab",
                 "ph_destab", "paper_full"),
                ("L1sec_phstab_minus_phdestab_LATE", "ph_stab",
                 "ph_destab", "paper_late"),
                ("L2_random_minus_noop_FULL", "random", "noop",
                 "paper_full")):
            d, lo, hi = boot_pairs(cu, a, b, key)
            entry["tests"][name] = {"diff": d, "ci": [lo, hi],
                                    "excludes0": bool(lo > 0
                                                      or hi < 0)}
        pf = np.array([u["paper_full"] for u in cu])
        cl = np.array([u["code_late"] for u in cu])
        ok = np.isfinite(pf) & np.isfinite(cl)
        entry["paper_vs_code_pearson"] = float(
            np.corrcoef(pf[ok], cl[ok])[0, 1]) if ok.sum() > 3 \
            else None
        results[cand] = entry

        print(f"\n=== Phase L candidate {cand} ===")
        print(f"{'arm':9s} {'paper(21-60)':>13s} {'paper(41-60)':>13s}")
        for arm in ARMS:
            a = entry["arms"][arm]
            print(f"{arm:9s} {a['paper_full']:13.4f} "
                  f"{a['paper_late']:13.4f}")
        for k, v in entry["tests"].items():
            print(f"{k}: {v['diff']:+.4f} "
                  f"CI [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
                  + ("  *" if v["excludes0"] else ""))
        print(f"paper-vs-code Pearson: {entry['paper_vs_code_pearson']}")

    with open(os.path.join(OUT, "phir_paper_results.json"),
              "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "phir_paper_results.json"))


if __name__ == "__main__":
    main()
